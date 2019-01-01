#include "utils.h"

std::string get_basedir(const std::string& filename)
{
    auto pos = filename.find_last_of("/\\");
    if(pos != std::string::npos) {
        return filename.substr(0, pos);
    }
    return "";
}
