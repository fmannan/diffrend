#version 110

#define MAX_NUM_LIGHTS 8

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 cam_pos;

uniform vec3 ambient;
uniform vec4 light_pos[MAX_NUM_LIGHTS];
uniform vec3 light_color[MAX_NUM_LIGHTS];  // emission
uniform vec3 light_attenuation[MAX_NUM_LIGHTS]; // attenuation coeffs constant, linear, quadratic
uniform int num_lights;

attribute vec3 position;
attribute vec3 normal;
attribute vec3 albedo;  // vertex color
attribute vec3 coeffs;

varying vec4 frag_position;
varying vec4 frag_normal;
varying vec3 frag_albedo;
varying vec3 frag_coeffs;


void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    frag_position = clamp(gl_Position, 0.0, 1.0);
    frag_normal = model * vec4(normal, 1.0); // TODO
    frag_albedo = albedo;
    frag_coeffs = coeffs;
}
