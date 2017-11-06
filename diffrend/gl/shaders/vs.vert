#version 120
in vec3 position;
uniform mat4 model_view;
void main()
{
   gl_Position = model_view * vec4(position, 1.0);
}
