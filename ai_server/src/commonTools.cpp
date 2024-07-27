#include "commonTools.h"

std::string to_string_with_truncation(float value, int precision) {
    int factor = std::pow(10, precision);
    value = std::floor(value * factor) / factor;

    std::string str = std::to_string(value);
    size_t dot_pos = str.find('.');
    if (dot_pos != std::string::npos && str.size() > dot_pos + precision + 1) {
        str.erase(dot_pos + precision + 1);
    }
    return str;
}