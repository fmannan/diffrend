#version 120

in vec3 v_albedo;
in vec3 v_normal;
in vec3 v_pos;

void main()
{
   gl_FragColor = vec4(v_pos.xyz, 1.0f);
}
