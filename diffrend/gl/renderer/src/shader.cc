#include <iostream>
#include <sstream>
#include <fstream>
#include "shader.h"

std::string load_shader_code(const std::string& path)
{
    std::string code = "";
    std::ifstream shader_file(path.c_str(), std::ios::in);
    if(shader_file.is_open()) {
        std::stringstream ss;
        ss << shader_file.rdbuf();
        code = ss.str();
    } else {
        std::cout << "Error: Unable to read " << path << std::endl;
    }
    return code;
}

GLuint compile_shader(const std::string& shader,
    GLuint shader_id) 
{
    char const * ptr = shader.c_str();
    glShaderSource(shader_id, 1, &ptr, NULL);
    glCompileShader(shader_id);

    GLint res = GL_FALSE;
    int infoLogLength;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &res);
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &infoLogLength);
    if(infoLogLength > 0) {
        std::cout << "compile_shader error:" << std::endl;
        GLchar* pInfoLog = new GLchar[infoLogLength + 1];
        GLint length = 0;
        glGetShaderInfoLog(shader_id, infoLogLength, &length, pInfoLog);
        std::cout << pInfoLog << std::endl;
        delete [] pInfoLog;
    }
    return res;
}

GLuint LoadShaders(const std::string& vertex_file_path,
    const std::string& fragment_file_path)
{
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    std::string vs_code = load_shader_code(vertex_file_path);
    std::string fs_code = load_shader_code(fragment_file_path);

    // Compile the shader
    compile_shader(vs_code, VertexShaderID);
    compile_shader(fs_code, FragmentShaderID);

    GLuint progID = glCreateProgram();
    glAttachShader(progID, VertexShaderID);
    glAttachShader(progID, FragmentShaderID);
    glLinkProgram(progID);

    // Check
    //...

    glDetachShader(progID, VertexShaderID);
    glDetachShader(progID, FragmentShaderID);

    return progID;
}
