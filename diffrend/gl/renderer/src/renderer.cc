#include <iostream>
#include <glad/glad.h>

#include "renderer.h"
#include <stb/stb_image_write.h>
#include <fstream>
#include <npy/npy.hpp>
#include "SPSC_LockFreeQueue.hpp"
#include <thread>

template<typename T>
struct StorageData {
    struct Data {
        T* mData;
        int mWidth;
        int mHeight;
        int mChannels;

        Data(const T* const data, int width, int height, int channels):
            mWidth(width), mHeight(height), mChannels(channels) {
                mData = new T[mWidth * mHeight * mChannels];
                memcpy(mData, data, sizeof(T) * mWidth * mHeight * mChannels);
            }
    };
    enum DType { INVALID=0, IMAGE, POSITION, NORMAL };
    Data* image;
    Data* position;
    Data* normal;
    //std::string mFilename;

    //StorageData(std::string&& filename) { mFilename = filename; }
    void set_image(const T* const data, int width, int height, int channels, DType dtype) {
        Data** dst(nullptr);
        if(dtype == DType::IMAGE)
            dst = &image;
        else if(dtype == DType::POSITION) {
            dst = &position;
        } else if(dtype == DType::NORMAL) {
            dst = &normal;
        }
        *dst = new Data(data, width, height, channels);
    }

    void set_position(const T* const data, int width , int height, int channels) {
        position = new Data(data, width, height, channels);
    }
};

using MsgQueue = SPSC_LockFreeQueue<StorageData<float>, 3000>;
MsgQueue gOutputQueue;
std::atomic<bool> bTerminate;

void WriteFrames(const std::string& out_filename_prefix,
		 MsgQueue& output_queue,
		 std::atomic<bool>& terminate)
{
  int frame_count = 0;
  char buffer[5000];
  try {
    while(true) {
      StorageData<float> data;
      if(!output_queue.pop(data)) {
        if(terminate) {
          std::cout << "Terminating WriteFrames..." << std::endl;
          break;
        }
        continue;
      }
      snprintf(buffer, sizeof(buffer), "%s/frame_%08d.tiff", out_filename_prefix.c_str(), frame_count);
      //...
      ++frame_count;
    }
  } catch(std::exception& e) {
    std::cout << "Exception in WriteFrames " << e.what() << std::endl;
  }
}

/////
static void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

void GLRenderer::key_callback(GLFWwindow* window, int key,
			 int scancode, int action,
			 int mods) {
  if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

GLRenderer::~GLRenderer() {
    bTerminate = true;
    if(mRGBA) {
        free(mRGBA);
    }
    if(mBuffer) {
        free(mBuffer);
    }
    mFrameWriterThread.join();
    glfwDestroyWindow(mWindow);
    glfwTerminate();
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

    // setting up framebuffer
    glGenFramebuffers(1, &mFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, mFBO);
    
    glGenTextures(1, &mTexRGBA);

    glBindTexture(GL_TEXTURE_2D, mTexRGBA);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mTexRGBA, 0);

    glGenTextures(1, &mTexPosition);

    glBindTexture(GL_TEXTURE_2D, mTexPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, mTexPosition, 0);

    glGenTextures(1, &mTexNormal);

    glBindTexture(GL_TEXTURE_2D, mTexNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, mTexNormal, 0);

    // depth attachment
    GLuint depth_buffer;
    glGenRenderbuffers(1, &depth_buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, mWidth, mHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);

    GLenum draw_buffers[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2  };
    glDrawBuffers(3, draw_buffers);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "Framebuffer setup failed" << std::endl;
        assert(false);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    mBuffer = static_cast<char*>(calloc(4, mWidth * mHeight));
    mRGBA = static_cast<float*>(calloc(4, mWidth * mHeight * sizeof(float)));
    bTerminate = false;
    mFrameWriterThread = std::thread(WriteFrames, mOutputDir,
          std::ref(gOutputQueue),
          std::ref(bTerminate));
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

void store_as_npy(const std::string& outfilename_prefix,
    int width, int height, int nchannels, float* data)
{
    unsigned long out_shape[] = {height, width, nchannels};
    std::vector<float> out_data(height * width * nchannels);
    memcpy(&out_data[0], data, sizeof(float) * height * width * nchannels);
    npy::SaveArrayAsNumpy(outfilename_prefix + ".npy",
        false, 3, out_shape, out_data);
}

void GLRenderer::render(const Camera* camera, const std::string& outfilename) {
    /**
     * Activate shader program
     * set camera and global transformations
     * for each object
     *   set transformation
     *   obj.render()
     * Write buffer to file
     */
    // set the FBO
    glBindFramebuffer(GL_FRAMEBUFFER, mFBO);
    glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        mScene->render(camera);

        // TODO: WRITE TO WRITER QUEUE
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_FLOAT, mRGBA);
        store_as_npy(mOutputDir + outfilename, mWidth, mHeight, 4, mRGBA);
        {
        std::ofstream outfile((mOutputDir + outfilename + ".dat").c_str(), std::ios::out | std::ios::binary);
        outfile.write((const char*) mRGBA, mWidth * mHeight * 4 * sizeof(float));
        }
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_FLOAT, mRGBA);
        store_as_npy(mOutputDir + outfilename + "_pos", mWidth, mHeight, 4, mRGBA);
        {
        std::ofstream outfile((mOutputDir + outfilename + "_pos.dat").c_str(), std::ios::out | std::ios::binary);
        outfile.write((const char*) mRGBA, mWidth * mHeight * 4 * sizeof(float));
        }
        glReadBuffer(GL_COLOR_ATTACHMENT2);
        glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_FLOAT, mRGBA);
        store_as_npy(mOutputDir + outfilename + "_normal", mWidth, mHeight, 4, mRGBA);
        {
        std::ofstream outfile((mOutputDir + outfilename + "_normal.dat").c_str(), std::ios::out | std::ios::binary);
        outfile.write((const char*) mRGBA, mWidth * mHeight * 4 * sizeof(float));
        }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mScene->render(camera);

    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, mBuffer);
    // Write image Y-flipped because OpenGL
    stbi_write_png((mOutputDir + outfilename + ".png").c_str(),
                   mWidth, mHeight, 4,
                   mBuffer + (mWidth * 4 * (mHeight - 1)),
                   -mWidth * 4);
}

void GLRenderer::updateCamera(const Camera& camera) {

}
