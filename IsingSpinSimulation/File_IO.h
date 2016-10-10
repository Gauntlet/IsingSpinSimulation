#ifndef FILE_IO_H
#define FILE_IO_H

#include <fstream>
#include <iostream>
#include "zlib.h"
#include "KDetails.h"

namespace kspace
{
	namespace FILEIO {

		/*///////////////////////////////////////////////////////
		FileHandle was taken from stackoverflow answer about RAII
		http://stackoverflow.com/a/713773.
		After test there was minimal overhead for wrapping
		FILE* within a class.
		*////////////////////////////////////////////////////////
		class FileHandle
		{
		private:
			FILE* file;
		public:
			explicit FileHandle( std::string fname, char* Mode )
			{
				file = fopen( fname.c_str(), Mode );
				if ( !file )
				{
					throw "File failed to open";
				}
			}

			~FileHandle()
			{
				if ( file )
				{
					fclose( file );
				}
			}

			FILE* operator()() const
			{
				return file;
			}

			//Remove the default copy constructors
			FileHandle( const FileHandle&& ) = delete;
			FileHandle& operator=( const FileHandle&& ) = delete;

			//Define move constructors for the file pointer.
			FileHandle( FileHandle&& that )
			{
				file = that.file;
				that.file = nullptr;
			}

			FileHandle& operator=( FileHandle&& that )
			{
				file = that.file;
				that.file = nullptr;
				return *this;
			}

			void close()
			{
				fclose( file );
				file = nullptr;
			}
		};

		


	}
}
#endif