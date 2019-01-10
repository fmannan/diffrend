#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "camera.h"

Camera::Camera(const Json::Value& camera_spec) {
    mPos = glm::vec3(camera_spec["eye"][0].asFloat(), camera_spec["eye"][1].asFloat(), camera_spec["eye"][2].asFloat());
    mAt = glm::vec3(camera_spec["at"][0].asFloat(), camera_spec["at"][1].asFloat(), camera_spec["at"][2].asFloat());
    mUp = glm::vec3(camera_spec["up"][0].asFloat(), camera_spec["up"][1].asFloat(), camera_spec["up"][2].asFloat());
    mFovy = camera_spec["fovy"].asFloat();
    mFocalLength = camera_spec["focal_length"].asFloat();
    mNear = camera_spec["near"].asFloat();
    mFar = camera_spec["far"].asFloat();

    for(int i = 0; i < 4; i++) {
        mViewport[i] = camera_spec["viewport"][i].asInt();
    }

    std::cout << str() << std::endl;
}

std::string Camera::str() const {
    std::stringstream ss;
    ss << "Camera configuration:\n";
    ss << "eye : [" << mPos[0] << "," << mPos[1] << "," << mPos[2] << "]\n";
    ss << "at  : [" << mAt[0] << "," << mAt[1] << "," << mAt[2] << "]\n";
    ss << "up  : [" << mUp[0] << "," << mUp[1] << "," << mUp[2] << "]\n";
    ss << "near: " << mNear << "\n";
    ss << "far : " << mFar << "\n";
    ss << "fovy: " << mFovy << "\n";
    ss << "focal_length: " << mFocalLength << "\n";
    ss << "viewport  : [" << mViewport[0] << "," << mViewport[1] << "," << mViewport[2] << "," << mViewport[3] << "]\n";
    ss << "\n";
    return ss.str();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(mPos, mAt, mUp);
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(mFovy, getAspectRatio(), mNear, mFar);
}

glm::mat4 Camera::getViewProjectionMatrix() const {
    return getProjectionMatrix() * getViewMatrix();
}

CameraTrajectory::CameraTrajectory(const std::string& trajectory_filename)
{
    std::ifstream ifs(trajectory_filename);

    std::cout << "Camera Trajectory file: " << trajectory_filename << std::endl;
    Json::CharReaderBuilder reader;
    Json::Value obj;
    std::string json_err;
    Json::parseFromStream(reader, ifs, &obj, &json_err);
    loadFromJson(obj);
}

CameraTrajectory::CameraTrajectory(const Json::Value& trajectory_spec)
{
    loadFromJson(trajectory_spec);
}

const Camera* CameraTrajectory::getNext(bool repeat)
{
    if(mCurrentTrajectoryId >= mCameras.size())
        return nullptr;

    const Camera* curr_cam = mCameras[mCurrentTrajectoryId++];
    if(repeat) {
        mCurrentTrajectoryId %= mCameras.size();
    }
    return curr_cam;
}

void CameraTrajectory::loadFromJson(const Json::Value& trajectory_spec)
{
    for(int i = 0; i < trajectory_spec.size(); i++) {
        std::cout << trajectory_spec[i] << std::endl;
        mCameras.push_back(new Camera(trajectory_spec[i]));
    }
    mCurrentTrajectoryId = 0;
}

std::pair<const Camera*, std::string> CameraTrajectory::getNextCameraAndFilename()
{
    char buffer[2048];
    sprintf(buffer, "im_%07d", mCurrentTrajectoryId);
    const Camera* cam = getNext(false);
    return std::make_pair(cam, std::string(buffer));
}
