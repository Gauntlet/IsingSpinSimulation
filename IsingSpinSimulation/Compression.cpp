#include "File_IO.h"


using namespace kspace::FILEIO;

Result Compression::deflate( const ArrayHandle& uncompressed_data )
{
	ArrayHandle	compressed_data( uncompressed_data.const_size()*1.1 + 12 );
	
	int return_code = compress( compressed_data.data(), &compressed_data.size(), uncompressed_data.data(), uncompressed_data.const_size() );

	switch ( return_code )
	{
	case Z_MEM_ERROR:
		throw std::runtime_error( "Out of memory while compressing data!" );
		break;
	case Z_BUF_ERROR:
		throw std::runtime_error( "Compress output buffer was not large enough!" );
		break;
	}

	Details details;
	details.uncompressed_data_size = uncompressed_data.const_size();
	details.bitpacked_data_size = uncompressed_data.const_size();
	details.compressed_data_size = compressed_data.const_size();
	details.bits.zerobit = 0;
	details.bits.unitbit = 0;

	return Result( compressed_data, details );
}

Result Compression::deflate( const ArrayHandle& uncompressed_data, const Bits& bits )
{
	//First bitpack the data.
	ArrayHandle bitpacked_data = bit_pack( uncompressed_data, bits );
	//Now compress the bitpacked data.
	Result compressed_data = deflate( bitpacked_data );
	//Change the values of uncompressed_data_size and bits in the details structure to the correct values
	compressed_data.details.uncompressed_data_size = uncompressed_data.const_size();
	compressed_data.details.bits = bits;

	return compressed_data;
}

ArrayHandle Compression::inflate( const ArrayHandle& compressed_data, const Details &details )
{
	//First decompress the data into its unpacked state.
	ArrayHandle unpacked_data( details.bitpacked_data_size );
	int return_code = uncompress( unpacked_data.data(), &unpacked_data.size(), compressed_data.data(), compressed_data.const_size() );

	switch ( return_code )
	{
	case Z_MEM_ERROR:
		throw std::runtime_error( "Out of memory while uncompressing data!" );
		break;
	case Z_BUF_ERROR:
		throw std::runtime_error( "Uncompress output buffer was not large enough!" );
		break;
	}
	
	//Check to see if the data has been bit packed.
	if ( details.bits.zerobit != details.bits.unitbit )
	{
		//Unpack the data and return it.
		ArrayHandle uncompressed_data = bit_unpack( unpacked_data, details );
		return uncompressed_data;
	}

	//The data was never packed before compression so return unpacked_data.
	return unpacked_data;
}


ArrayHandle Compression::bit_pack( const ArrayHandle &uncompressed_data, const Bits &bits)
{
	ArrayHandle bitpacked_data( ( uncompressed_data.const_size() / 8 ) + 1 );

	for ( size_t i = 0; i < uncompressed_data.const_size(); ++i )
	{
		size_t j = i / 8;
		size_t k = i % 8;

		if ( uncompressed_data( i ) > 0 )
		{
			bitpacked_data[ j ] |= 1 << k;
		}
	}

	return bitpacked_data;
}

ArrayHandle Compression::bit_unpack( const ArrayHandle &bitpacked_data, const Details &details )
{
	ArrayHandle uncompressed_data( details.uncompressed_data_size);

	for ( size_t i = 0; i < bitpacked_data.const_size(); ++i )
	{
		for ( size_t j = 0; j < 8; ++j )
		{
			size_t k = i * 8 + j;
			if ( k < uncompressed_data.const_size() )
			{
				if ( 1 == ( ( bitpacked_data( i ) >> j ) & 1 ) )
				{
					uncompressed_data[ k ] = details.bits.unitbit;
				}
				else
				{
					uncompressed_data[ k ] = details.bits.zerobit;
				}
			}
		}
	}

	return uncompressed_data;
}

