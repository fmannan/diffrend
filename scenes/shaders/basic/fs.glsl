#version 110

varying vec4 frag_position;
varying vec3 frag_albedo;

void main() {
    gl_FragColor = frag_position; //vec4(frag_albedo, 1.0);
}
