#ifndef FILE_IO_H
#define FILE_IO_H

#include <fstream>

namespace kspace
{
	namespace FILEIO {
#include "zlib.h"

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

		class ArrayHandle
		{
		private:
			Byte* _data;
			uLong _size;
		public:
			explicit ArrayHandle(const size_t size)
			{
				_data = new Byte[ size ]();
				_size = size;
			}

			~ArrayHandle()
			{
				if ( _data )
				{
					delete[] _data;
				}
			}

			Byte* data() const
			{
				return _data;
			}

			uLong& size()
			{
				return _size;
			}

			uLong const_size() const
			{
				return _size;
			}

			Byte& operator[]( const size_t i )
			{
				return _data[ i ];
			}

			Byte operator()( const size_t i ) const
			{
				return _data[ i ];
			}

			//Remove the default copy constructors
			ArrayHandle( const ArrayHandle&& ) = delete;
			ArrayHandle& operator=( const ArrayHandle&& ) = delete;

			//Define move constructors for the file pointer.
			ArrayHandle( ArrayHandle&& that )
			{
				_data = that._data;
				that._data= nullptr;
			}

			ArrayHandle& operator=( ArrayHandle&& that )
			{
				_data = that._data;
				that._data= nullptr;
				return *this;
			}
		};

		class Compression
		{		
		public:

			struct Bits
			{
				Byte zerobit;
				Byte unitbit;
			};

			struct Details
			{
				uLong uncompressed_data_size;
				uLong bitpacked_data_size;
				uLong compressed_data_size;

				Bits bits;
			};

			struct Result
			{
				ArrayHandle data;
				Details details;
			};

			Result deflate( const ArrayHandle& uncompressed_data);
			Result deflate( const ArrayHandle& uncompressed_data, const Bits& bits);

			ArrayHandle inflate( const ArrayHandle& compressed_data, const Details& details );

		protected:
			ArrayHandle bit_pack( const ArrayHandle &uncompressed_data, const Bits &bits);
			ArrayHandle bit_unpack( const ArrayHandle &bitpacked_data, const Details &details );
		};
	}
}
#endif