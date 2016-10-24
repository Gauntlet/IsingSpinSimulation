#ifndef COMPRESSION_H
#define COMPRESSION_H

#include "File_IO.h"
#include "Array.h"

namespace kspace
{
	namespace FILEIO
	{
		/**
		* This class is a wrapper for compressing and uncompressing data stored in an Array.
		*/
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

				Compressed_Data();
				Compressed_Data( Compressed_Data&& that ) : data( std::move( that.data ) ), details( that.details ) {};
				Compressed_Data( Array<std::uint8_t> &data, Details &details ) : data( std::move( data ) ), details( details ) {};
			};

			Compressed_Data deflate( Array<std::uint8_t> const & uncompressed_data );
			Compressed_Data deflate( Array<std::uint8_t> const & uncompressed_data, Bits const & bits );

			Array<std::uint8_t> inflate( Array<std::uint8_t> const & compressed_data, Details const & details );

		protected:
			Array<std::uint8_t> bit_pack( Array<std::uint8_t> const & uncompressed_data, Bits const & bits );
			Array<std::uint8_t> bit_unpack( Array<std::uint8_t> const & bitpacked_data, Details const & details );
		};
	}
}

#endif