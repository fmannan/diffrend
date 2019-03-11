#version 110

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

attribute vec3 albedo;  // vertex color
attribute vec3 position;

varying vec4 frag_position;
varying vec3 frag_albedo;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    frag_position = clamp(gl_Position, 0.0, 1.0);
    frag_albedo = albedo;
}
