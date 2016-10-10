#include "Compression.h"
#include "File_IO.h"

using namespace kspace;
using namespace FILEIO;

Compression::Compressed_Data Compression::deflate( const ArrayHandle& uncompressed_data )
{
	ArrayHandle	compressed_data( uncompressed_data.get.size()*1.1 + 12 );
	
	int return_code = compress( compressed_data.set.data(), compressed_data.set.size(), uncompressed_data.get.data(), uncompressed_data.get.size() );

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
	details.uncompressed_data_size = uncompressed_data.get.size();
	details.bitpacked_data_size = uncompressed_data.get.size();
	details.compressed_data_size = compressed_data.get.size();
	details.bits.zerobit = 0;
	details.bits.unitbit = 0;

	return Compression::Compressed_Data( compressed_data, details );
}

Compression::Compressed_Data Compression::deflate( const ArrayHandle& uncompressed_data, const Compression::Bits& bits )
{
	//First bitpack the data.
	ArrayHandle bitpacked_data = bit_pack( uncompressed_data, bits );
	//Now compress the bitpacked data.
	Compressed_Data compressed_data = deflate( bitpacked_data );
	//Change the values of uncompressed_data_size and bits in the details structure to the correct values
	compressed_data.details.uncompressed_data_size = uncompressed_data.get.size();
	compressed_data.details.bits = bits;

	return compressed_data;
}

ArrayHandle Compression::inflate( const ArrayHandle& compressed_data, const Compression::Details &details )
{
	//First decompress the data into its unpacked state.
	ArrayHandle uncompressed_data( details.bitpacked_data_size );
	int return_code = uncompress( uncompressed_data.set.data(), uncompressed_data.set.size(), compressed_data.get.data(), compressed_data.get.size() );

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
	if ( details.bitpacked_data_size > 0 )
	{
		Details tmpdet = details;

		if ( details.bits.unitbit == details.bits.zerobit )
		{
			tmpdet.bits.unitbit = 1;
			tmpdet.bits.zerobit = 0;
		}

		//Unpack the data and return it.
		ArrayHandle unpacked_data = bit_unpack( unpacked_data, tmpdet );
		return unpacked_data;
	}

	//The data was never packed before compression so return unpacked_data.
	return uncompressed_data;
}


ArrayHandle Compression::bit_pack( const ArrayHandle &uncompressed_data, const Compression::Bits &bits )
{
	ArrayHandle bitpacked_data( ( uncompressed_data.get.size() / 8 ) + 1 );

	for ( std::size_t i = 0; i < uncompressed_data.get.size(); ++i )
	{
		std::size_t j = i / 8;
		std::size_t k = i % 8;

		if ( uncompressed_data.get.data( i ) > 0 )
		{
			bitpacked_data.set.data[ j ] |= 1 << k;
		}
	}

	return bitpacked_data;
}

ArrayHandle Compression::bit_unpack( const ArrayHandle &bitpacked_data, const Compression::Details &details )
{
	ArrayHandle uncompressed_data( details.uncompressed_data_size);

	for ( std::size_t i = 0; i < bitpacked_data.get.size(); ++i )
	{
		for ( std::size_t j = 0; j < 8; ++j )
		{
			std::size_t k = i * 8 + j;
			if ( k < uncompressed_data.get.size() )
			{
				const std::uint8_t byte = *bitpacked_data.get.data( i );
				if ( 1 == ( ( byte >> j ) & 1 ) )
				{
					uncompressed_data.set.data[ k ] = details.bits.unitbit;
				}
				else
				{
					uncompressed_data.set.data[ k ] = details.bits.zerobit;
				}
			}
		}
	}

	return uncompressed_data;
}

