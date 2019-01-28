#version 330

#define MAX_NUM_LIGHTS 8

uniform vec3 cam_pos;

uniform mat4 view;

uniform vec3 ambient;
uniform vec4 light_pos[MAX_NUM_LIGHTS];
uniform vec3 light_color[MAX_NUM_LIGHTS];  // emission
uniform vec3 light_attenuation[MAX_NUM_LIGHTS]; // attenuation coeffs constant, linear, quadratic
//uniform int num_lights;

in vec4 frag_position;
in vec4 frag_normal;
in vec3 frag_albedo;
in vec3 frag_coeffs;

layout(location=0) out vec4 color;
layout(location=1) out vec4 pos;
layout(location=2) out vec4 normal;

vec4 get_cam_dir_normal()
{
    // Flip per-fragment normals if needed based on the camera direction
    vec3 surface_normal = normalize(frag_normal.xyz);
    vec4 cam_pos_viewspace = view * vec4(cam_pos, 1.0);
    vec3 cam_dir = normalize(cam_pos_viewspace.xyz - frag_position.xyz);
    float dot_prod = dot(cam_dir, surface_normal);
    float sgn = sign(dot_prod);
    return sgn * vec4(surface_normal, 0.0);
}

void main() {
    pos = frag_position;
    normal = get_cam_dir_normal();
    vec4 light_irradiance = vec4(0.0);

    for(int i = 0; i < MAX_NUM_LIGHTS; i++) {
        vec4 lpos = view * light_pos[i];
        vec4 light_dir = lpos - pos;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);
        float divisor = (light_attenuation[i].x + light_dist * light_attenuation[i].y + light_dist * light_dist * light_attenuation[i].z);
        if(abs(divisor) < 1e-8)
            divisor = 1.0;
        float att_factor = 1.0 / divisor;
        light_irradiance += vec4(light_color[i], 1.0) * dot(normal, light_dir) * att_factor;
    }
    vec4 clr = clamp(vec4(frag_albedo, 1.0) * light_irradiance + vec4(ambient * 10.0, 1.0), 0.0, 1.0);

    color = vec4(clr.xyz, 1.0);
}
