#include "image.hpp"
#include <exception>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

Image::Image( const int p_width, const int p_height, const int p_nbChannels )
    : _width( p_width ), _height( p_height ), _nbChannels( p_nbChannels )
{
    _pixels = new unsigned char[ _width * _height * _nbChannels ];
    memset( _pixels, 0, sizeof(_pixels) );
}

Image::~Image() { delete[] _pixels; }

void Image::load( const std::string & p_path )
{
    _pixels = stbi_load( p_path.c_str(), &_width, &_height, &_nbChannels, 0 );
    if ( _pixels == nullptr )
    {
        std::string msg = "Failed to load image: " + p_path + "\n" + stbi_failure_reason();
        throw std::runtime_error( msg.c_str() );
    }
}

void Image::save( const std::string & p_path ) const
{
    stbi_write_png( p_path.c_str(), _width, _height, _nbChannels, _pixels, _width * _nbChannels );
}
