#pragma once

#include <string>
#include <json/json.h>
#include <glm/glm.hpp>

class Camera {
public:
    Camera(const Json::Value& camera_spec);
    Camera(glm::vec3 pos, glm::vec3 lookat, glm::vec3 up, float focal_length, float fovy);
    Camera(float focal_length, float fovy);

    void setCameraParameters(glm::vec3 pos, glm::vec3 lookat, glm::vec3 up, float focal_length, float fovy);

    glm::mat4 getViewMatrix();
    glm::mat4 getProjectionMatrix();
    glm::mat4 getViewProjectionMatrix();

    float getAspectRatio() { return getWidth() / getHeight(); }
    int getWidth() { return mViewport[2] - mViewport[0]; }
    int getHeight() { return mViewport[3] - mViewport[1]; }

    std::string str() const;
private:
    glm::vec3 mPos;
    glm::vec3 mUp;
    glm::vec3 mAt;
    glm::vec3 mWorldUp;
    float mFovy;
    float mFocalLength;
    float mNear, mFar;
    int mViewport[4];
};

