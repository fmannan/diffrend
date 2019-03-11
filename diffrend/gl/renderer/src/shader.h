#pragma once

#include <string>
#include <glad/glad.h>

std::string load_shader_code(const std::string& path);
GLuint compile_shader(const std::string& shader, GLuint shader_id);
GLuint LoadShaders(const std::string& vertex_file_path,
    const std::string& fragment_file_path);


class GLShader {

};

class GLProgram {

};
