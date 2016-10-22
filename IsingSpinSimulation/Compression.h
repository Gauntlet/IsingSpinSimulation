#ifndef COMPRESSION_H
#define COMPRESSION_H

#include "File_IO.h"
#include "Array.h"

namespace kspace
{
	namespace FILEIO
	{
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

			struct Compressed_Data
			{
				Array<std::uint8_t> data;
				Details details;

				Compressed_Data( Compressed_Data&& that ) : data( std::move( that.data ) ), details( that.details ) {};
				Compressed_Data( Array<std::uint8_t> &data, Details &details ) : data( std::move( data ) ), details( details ) {};
			};

			Compressed_Data deflate( const Array<std::uint8_t>& uncompressed_data );
			Compressed_Data deflate( const Array<std::uint8_t>& uncompressed_data, const Bits& bits );

			Array<std::uint8_t> inflate( const Array<std::uint8_t>& compressed_data, const Details& details );

		protected:
			Array<std::uint8_t> bit_pack( const Array<std::uint8_t> &uncompressed_data, const Bits &bits );
			Array<std::uint8_t> bit_unpack( const Array<std::uint8_t> &bitpacked_data, const Details &details );
		};
	}
}

#endif