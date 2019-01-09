#pragma once

#include <string>
#include <vector>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glad/glad.h>


struct Object {
public:
    virtual glm::mat4 get_transformation() const = 0;
    virtual void set_transformations(glm::vec3 translate, glm::mat4 rotate, glm::vec3 scale) = 0;
};

using GLSLVarMap = std::map<std::string, GLuint>;

class GLRenderableObject: public Object {
    // Objects that can be rendered using OpenGL
public:
    virtual void setup(GLSLVarMap& var_map) = 0;
    virtual void render(GLint model_matrix_location) = 0;
};

struct Material {
    glm::vec3 albedo;
    glm::vec3 coeffs; // kd, ks, specular highight N in (cos(theta))^N
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    Material material;
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
    void set_transformations(glm::vec3 translate, glm::mat4 rotate, glm::vec3 scale) override
    {}

    glm::mat4 get_transformation() const override
    {
        return glm::mat4(1.0);
    }

    void render(GLint model_matrix_location) override {
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
    TriangleMesh(const std::string& obj_filename,
        Material mat, glm::vec3 translate=glm::vec3(0.0),
        glm::mat4 rotate=glm::mat4(1.0),
        glm::vec3 scale=glm::vec3(1.0))
        : mTranslation(translate), mRotation(rotate),
        mScale(scale) {
        loadObj(obj_filename, mat);
    }

    void set_transformations(glm::vec3 translate, glm::mat4 rotate, glm::vec3 scale) override;
    glm::mat4 get_transformation() const override { 
        return glm::translate(glm::mat4(1.0), mTranslation) * mRotation * glm::scale(glm::mat4(1.0), mScale);
    }
    void setup(GLSLVarMap& var_map) override;
    void render(GLint model_matrix_location) override;
private:
    GLuint mVAO;
    GLuint mVBO;
    //GLuint mIBO;
    std::vector<Vertex> mBuffer;
    //std::vector<glm::vec3> mVertices;
    //std::vector<glm::vec3> mNormals;
    //std::vector<glm::vec2> mTexCoords;
    //std::vector<glm::ivec3> mIndices;   // vertex indices forming a face

    Material mMaterial; // single material for all faces
    
    glm::vec3 mTranslation;
    glm::mat4 mRotation;
    glm::vec3 mScale;

    void loadObj(const std::string& filename, Material mat);
};

