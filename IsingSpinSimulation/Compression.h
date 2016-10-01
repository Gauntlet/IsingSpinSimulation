#ifndef COMPRESSION_H
#define COMPRESSION_H

#include "File_IO.h"

namespace kspace {
	namespace FILEIO {
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

				Result( Result&& that ) : data( std::move( that.data ) ), details( that.details ) {};
				Result( ArrayHandle &data, Details &details ) : data( std::move( data ) ), details( details ) {};
			};

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