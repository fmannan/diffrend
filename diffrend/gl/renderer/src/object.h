#pragma once

#include <string>
#include <vector>
#include <map>

#include <glm/glm.hpp>
#include <glad/glad.h>


struct Object {

};

using GLSLVarMap = std::map<std::string, GLuint>;

class GLRenderableObject: public Object {
    // Objects that can be rendered using OpenGL
public:
    virtual void setup(GLSLVarMap& var_map) = 0;
    virtual void render() = 0;
};

struct Material {
    glm::vec3 albedo;
    glm::vec3 coeffs; // kd, ks, theta
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    Material mat;
};

class TestTriangle : public GLRenderableObject {
public:
    TestTriangle() {
        // constructor only for creating the geometry
    }

    void setup(GLSLVarMap& var_map) override {
        GLuint vpos_location = var_map["position"];
        GLuint vcol_location = var_map["albedo"];
        const struct {
            float x, y, z;
            float r, g, b;
        } vertices[3] = {
            { -0.6f, -0.4f, 0.0f, 1.f, 0.f, 0.f },
            {  0.6f, -0.4f, 0.0f, 0.f, 1.f, 0.f },
            {   0.f,  0.6f, 0.0f, 0.f, 0.f, 1.f }
        };
        // create vao??
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        // Create VBO
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(vpos_location);
        glVertexAttribPointer(vpos_location, 3, GL_FLOAT, GL_FALSE,
                            sizeof(vertices[0]), (void*) 0);
        glEnableVertexAttribArray(vcol_location);
        glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE,
                            sizeof(vertices[0]), (void*) (sizeof(float) * 3));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void render() override {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
    }

private:
    GLuint vbo;
    GLuint vao;
};

class Triangle {
    // A triangle face
public:
    Triangle(glm::vec4 vertices[3], glm::vec4 normal);

private:
    glm::vec4 mVertices[3];
    glm::vec4 normal;
};

class TriangleMesh: public GLRenderableObject {
    // Renderable triangle mesh
    // List of vertices and faces of triangles
public:
    TriangleMesh(const std::string& obj_filename) {
        loadObj(obj_filename);
    }

    void setup(GLSLVarMap& var_map) override;
    void render() override;
private:
    GLuint mVao;
    GLuint mVbo;
    GLuint mIbo;
    std::vector<float> mBuffer;
    std::vector<glm::vec3> mVertices;
    std::vector<glm::ivec3> mIndices;   // vertex indices forming a face

    void loadObj(const std::string& filename);
};

