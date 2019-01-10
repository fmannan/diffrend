#version 110

#define MAX_NUM_LIGHTS 8

uniform vec3 cam_pos;

uniform vec3 ambient;
uniform vec4 light_pos[MAX_NUM_LIGHTS];
uniform vec3 light_color[MAX_NUM_LIGHTS];  // emission
uniform vec3 light_attenuation[MAX_NUM_LIGHTS]; // attenuation coeffs constant, linear, quadratic
//uniform int num_lights;

varying vec4 frag_position;
varying vec4 frag_normal;
varying vec3 frag_albedo;
varying vec3 frag_coeffs;

void main() {
    //gl_FragColor = frag_normal; //frag_position; //vec4(frag_albedo, 1.0);
    vec4 light_irradiance = vec4(light_color[0], 1.0) * dot(normalize(frag_normal), normalize(light_pos[0] - frag_position));
    for(int i = 1; i < MAX_NUM_LIGHTS; i++) {
        light_irradiance += vec4(light_color[i], 1.0) * dot(normalize(frag_normal), normalize(light_pos[i] - frag_position));
    }
    vec4 color = clamp(vec4(frag_albedo, 1.0) * light_irradiance + vec4(ambient * 10.0, 1.0), 0.0, 1.0);
    gl_FragColor = vec4(color.xyz, 1.0);
}
