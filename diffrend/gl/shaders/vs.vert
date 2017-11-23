#version 120

in vec3 position;
in vec3 normal;
in vec3 albedo;

varying vec3 v_albedo;
varying vec3 v_normal;
varying vec3 v_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
   gl_Position = projection * view * model * vec4(position, 1.0);
   vec4 pos = view * model * vec4(position, 1.0);
   v_pos = pos.xyz / pos.w;
   v_albedo = albedo;
   v_normal = normal;
}
