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

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getViewProjectionMatrix() const;

    float getAspectRatio() const { return getWidth() / getHeight(); }
    int getWidth() const { return mViewport[2] - mViewport[0]; }
    int getHeight() const { return mViewport[3] - mViewport[1]; }

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

class CameraTrajectory {
public:
    CameraTrajectory(const std::string& trajectory_filename);
    CameraTrajectory(const Json::Value& trajectory_spec);
    const Camera* getNext(bool repeat);
    std::pair<const Camera*, std::string> getNextCameraAndFilename();
private:
    std::vector<Camera*> mCameras;
    int mCurrentTrajectoryId;

    void loadFromJson(const Json::Value& trajectory_spec);
};
