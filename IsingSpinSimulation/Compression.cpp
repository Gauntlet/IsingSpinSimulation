#include "Compression.h"
#include "File_IO.h"

using namespace kspace;
using namespace FILEIO;

/**
* Compresses the passed array using the zlib library.
* @param uncompressed_array an Array<std::uint8_t> of uncompressed data.
* @return a Compressed_Data structure containing the compressed array and details.
*/
Compression::Compressed_Data Compression::deflate( Array<std::uint8_t> const & uncompressed_data )
{
	Compression::Compressed_Data cdata;

	if ( MemoryLocation::host == uncompressed_data.get.memory_location() )
	{
		Array<std::uint8_t> compressed_data( uncompressed_data.get.size()*1.1 + 12 );

		int return_code = compress( compressed_data.set.data_ptr(), &compressed_data.set.size(), uncompressed_data.get.data(), uncompressed_data.get.size() );

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

		cdata.data = std::move(compressed_data);
		cdata.details = details;
	}
	
	return cdata;
}

/**
* Bit packs the passed Array and then uses the zlib library to compress the bitpacked data.
* @param uncompressed_data an Array<std::uint8_t> of uncompressed data.
* @param bits contains the value of the 0 and 1 bits.
* @return a Compressed_Data structure containing the compressed array and details.
*/
Compression::Compressed_Data Compression::deflate( Array<std::uint8_t> const & uncompressed_data, Compression::Bits const & bits )
{
	//First bitpack the data.
	Array<std::uint8_t> bitpacked_data = bit_pack( uncompressed_data, bits );
	//Now compress the bitpacked data.
	Compressed_Data compressed_data = deflate( bitpacked_data );
	//Change the values of uncompressed_data_size and bits in the details structure to the correct values
	compressed_data.details.uncompressed_data_size = uncompressed_data.get.size();
	compressed_data.details.bits = bits;

	return compressed_data;
}

/**
* Uncompresses compressed data using the zlib library.
* @param compressed_data.
* @param details a struct that contains the sizes of the compressed, bitpacked and uncompressed data.
* @param uncompressed data.
*/
Array<std::uint8_t> Compression::inflate( Array<std::uint8_t> const & compressed_data, const Compression::Details &details )
{
	//First decompress the data into its unpacked state.
	Array<std::uint8_t> uncompressed_data( details.bitpacked_data_size );
	int return_code = uncompress( uncompressed_data.set.data_ptr(), &uncompressed_data.set.size(), compressed_data.get.data(), compressed_data.get.size() );

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
		Array<std::uint8_t> unpacked_data = bit_unpack( unpacked_data, tmpdet );
		return unpacked_data;
	}

	//The data was never packed before compression so return unpacked_data.
	return uncompressed_data;
}

/**
* Takes an array of binary data and packs them into bytes. Essentially this reduces the data size by a factor of 8 before
* any compression has occurred.
* @param uncompressed_data.
* @param bits.
* @return bitpacked data.
*/
Array<std::uint8_t> Compression::bit_pack( Array<std::uint8_t> const & uncompressed_data, Compression::Bits const & bits )
{
	Array<std::uint8_t> bitpacked_data( ( uncompressed_data.get.size() / 8 ) + 1 );

	for ( std::size_t i = 0; i < uncompressed_data.get.size(); ++i )
	{
		std::size_t j = i / 8;
		std::size_t k = i % 8;

		if ( uncompressed_data.get( i ) > 0 )
		{
			bitpacked_data.set.data[ j ] |= 1 << k;
		}
	}

	return bitpacked_data;
}

/**
* Unpacks an array of bitpacked data.
* @param bitpacked_data.
* @param details.
* @return unbitpacked data.
*/
Array<std::uint8_t> Compression::bit_unpack( Array<std::uint8_t> const & bitpacked_data, Compression::Details const & details )
{
	Array<std::uint8_t> uncompressed_data( details.uncompressed_data_size);

	for ( std::size_t i = 0; i < bitpacked_data.get.size(); ++i )
	{
		for ( std::size_t j = 0; j < 8; ++j )
		{
			std::size_t k = i * 8 + j;
			if ( k < uncompressed_data.get.size() )
			{
				const std::uint8_t byte = bitpacked_data.get( i );
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

