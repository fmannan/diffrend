#include <glad/glad.h>

#include "renderer.h"
#include <stb/stb_image_write.h>


static void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

void GLRenderer::key_callback(GLFWwindow* window, int key,
			 int scancode, int action,
			 int mods) {
  if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}


void GLRenderer::init() {
    glfwSetErrorCallback(error_callback);

    glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);

    if(!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    //glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    mWindow = glfwCreateWindow(mWidth, mHeight, "Render Server", NULL, NULL);

    if(!mWindow) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(mWindow);
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
    
    glfwSetWindowUserPointer(mWindow, this);
    glfwSetKeyCallback(mWindow, key_callback);
    // get version info
    const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString(GL_VERSION); // version as a string
    printf("Renderer: %s\n", renderer);
    printf("OpenGL version supported %s\n", version);
}

void GLRenderer::setupScene() {
    glfwGetFramebufferSize(mWindow, &mWidth, &mHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    //ratio = mWidth / (float) mHeight;
    mScene->setup();
    glViewport(0, 0, mWidth, mHeight);
}

void GLRenderer::render(Scene* scene) {
    mScene = scene;

    // Setup the resources on the GPU
    setupScene();

    // Render
    render();
}

void GLRenderer::render() {
    /**
     * Activate shader program
     * set camera and global transformations
     * for each object
     *   set transformation
     *   obj.render()
     * Write buffer to file
     */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mScene->render();

#if USE_NATIVE_OSMESA
    glfwGetOSMesaColorBuffer(mWindow, &mWidth, &mHeight, NULL, (void**) &buffer);
#else
    buffer = static_cast<char*>(calloc(4, mWidth * mHeight));
    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
#endif

    // Write image Y-flipped because OpenGL
    stbi_write_png("offscreen.png",
                   mWidth, mHeight, 4,
                   buffer + (mWidth * 4 * (mHeight - 1)),
                   -mWidth * 4);
}

void GLRenderer::updateCamera(const Camera& camera) {

}
