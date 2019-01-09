#include "utils.h"
#include <glm/glm.hpp>
#include <iostream>


std::string get_basedir(const std::string& filename)
{
    auto pos = filename.find_last_of("/\\");
    if(pos != std::string::npos) {
        return filename.substr(0, pos);
    }
    return "";
}

void print_mat(const glm::mat4& m) {
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }    
}