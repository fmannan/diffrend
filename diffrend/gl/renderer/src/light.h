#pragma once

#include <glm/glm.hpp>


class Light {
public:
    Light(glm::vec4 pos, glm::vec3 color);

    enum LightType {AMBIENT, DIRECTIONAL, POINT, SPOT};
private:
    glm::vec4 mPosition;
    glm::vec3 mColor;
};
