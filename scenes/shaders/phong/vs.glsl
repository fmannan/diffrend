#version 330

#define MAX_NUM_LIGHTS 8

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 inv_model_view;
uniform mat4 inv_model_view_transpose;
uniform vec3 cam_pos;

uniform vec3 ambient;
uniform vec4 light_pos[MAX_NUM_LIGHTS];
uniform vec3 light_color[MAX_NUM_LIGHTS];  // emission
uniform vec3 light_attenuation[MAX_NUM_LIGHTS]; // attenuation coeffs constant, linear, quadratic
uniform int num_lights;

in vec3 position;
in vec3 normal;
in vec3 albedo;  // vertex color
in vec3 coeffs;

// fragment params in view space
out vec4 frag_position;
out vec4 frag_normal;
out vec3 frag_albedo;
out vec3 frag_coeffs;


void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    frag_position = view * model * vec4(position, 1.0);
    frag_normal = normalize(inv_model_view_transpose * vec4(normal, 0.0));
    frag_albedo = albedo;
    frag_coeffs = coeffs;
}
