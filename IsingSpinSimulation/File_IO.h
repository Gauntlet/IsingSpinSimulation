#ifndef FILE_IO_H
#define FILE_IO_H

#include <fstream>
#include <iostream>

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
			//Remove the default copy constructors
			//ArrayHandle( const ArrayHandle& ) = delete;
			//ArrayHandle& operator=( const ArrayHandle& ) = delete;

			//Define move constructors for the file pointer.
			ArrayHandle( ArrayHandle&& that ) : _data( std::move( that._data ) ), _size( std::move( that._size ) ) 
			{
				that._data = nullptr;
				that._size = 0;
			};
			
			//ArrayHandle( ArrayHandle& that ) : _data( std::move( that._data ) ), _size( std::move( that._size ) ) {};

			ArrayHandle& operator=( ArrayHandle&& that )
			{
				_data = that._data;
				that._data = nullptr;
				that._size = 0;
				return *this;
			}

			ArrayHandle( const ArrayHandle& that ) = delete;
			ArrayHandle& operator=( const ArrayHandle& that ) = delete;

			ArrayHandle( const uLong size )
			{
				_data = new Byte[ size ]();
				_size = size;
			}


			explicit ArrayHandle( Byte*& data, const uLong size )
			{
				_data = data;
				_size = size;
				data = nullptr;
			}


			~ArrayHandle()
			{
				if ( nullptr != _data )
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
		};

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

			Result( Result&& that ) : data( std::move( that.data ) ), details( that.details ) {};
			Result( ArrayHandle &data, Details &details ) : data( std::move(data) ), details( details ) {};
		};

		class Compression
		{
		public:
			Result deflate( const ArrayHandle& uncompressed_data );
			Result deflate( const ArrayHandle& uncompressed_data, const Bits& bits );

			ArrayHandle inflate( const ArrayHandle& compressed_data, const Details& details );

		protected:
			ArrayHandle bit_pack( const ArrayHandle &uncompressed_data, const Bits &bits );
			ArrayHandle bit_unpack( const ArrayHandle &bitpacked_data, const Details &details );
		};
	}
}
#endif