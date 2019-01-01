#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#if USE_NATIVE_OSMESA
    #define GLFW_EXPOSE_NATIVE_OSMESA
    #include <GLFW/glfw3native.h>
#endif

#include "object.h"
#include "scene.h"
#include "camera.h"

class GLRenderer {
public:
    GLRenderer(const std::string& output_dir, int width, int height): mOutputDir(output_dir),
    mWidth(width), mHeight(mHeight) {
        init();
    }
    GLRenderer(Scene* scene, const std::string& output_dir): 
        mScene(scene), mOutputDir(output_dir) 
        {   
            mWidth = scene->getWidth();
            mHeight = scene->getHeight();
            init();
            setupScene();
        }
    ~GLRenderer() {
        glfwDestroyWindow(mWindow);
        glfwTerminate();
    }

    // Default rendering
    void render();

    // Render a given scene. Useful when the scene is updated
    void render(Scene* scene);

    // Render the current scene from a different viewpoint
    void render(Camera& camera);

    int shouldClose() { return glfwWindowShouldClose(mWindow); }
    void swapBuffers() { glfwSwapBuffers(mWindow);  }

    static void key_callback(GLFWwindow* window,
        int key, int scancode, int action,
        int mods);
private:
    std::string mOutputDir;
    Scene* mScene;
    GLFWwindow* mWindow;
    int mWidth, mHeight;
    char* buffer;

    void init();
    void setupScene();
    void updateCamera(const Camera& camera);
};