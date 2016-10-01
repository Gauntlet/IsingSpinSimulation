#include "Parameters.h"
using namespace kspace::GRAPH;

template <class RT, class T>
RT preserved_cast( T num )
{
	return *( (RT*) ( &num ) );
}

template <class T>
std::string num2str( const Parameters::NUMTYPE numtype, T num )
{
	switch ( numtype )
	{
	case Parameters::NUMTYPE::UINT8:
		std::uint8_t casted_num = preserved_cast<std::uint8_t>( num );
		break;

	case Parameters::NUMTYPE::UINT16:
		std::uint16_t casted_num = preserved_cast<std::uint16_t>( num );
		break;

	case Parameters::NUMTYPE::UINT32:
		std::uint32_t casted_num = preserved_cast<std::uint32_t>( num );
		break;

	case Parameters::NUMTYPE::UINT64:
		std::uint64_t casted_num = preserved_cast<std::uint64_t>( num );
		break;

	case Parameters::NUMTYPE::INT8:
		std::uint8_t casted_num = preserved_cast<std::uint8_t>( num );
		break;

	case Parameters::NUMTYPE::INT16:
		std::uint16_t casted_num = preserved_cast<std::uint16_t>( num );
		break;

	case Parameters::NUMTYPE::INT32:
		std::uint32_t casted_num = preserved_cast<std::uint32_t>( num );
		break;

	case Parameters::NUMTYPE::INT64:
		std::uint64_t casted_num = preserved_cast<std::uint64_t>( num );
		break;

	case Parameters::NUMTYPE::FLOAT32:
		float casted_num = preserved_cast<float>( num );
		break;

	case Parameters::NUMTYPE::FLOAT64:
		double casted_num = preserved_cast<double>( num );
		break;

	case Parameters::NUMTYPE::BOOLEAN:
		bool casted_num = preserved_cast<bool>( num );
		break;
	default:
		throw std::runtime_error
	}

	return std::to_string( casted_num );
}

template <class T>
Parameters::NUMTYPE to_numtype( T num )
{
	if ( typeid( std::uint8_t ).name() == typeid( T ).name() || typeid( std::uint8_t* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::UINT8;
	}
	else if ( typeid( std::uint16_t ).name() == typeid( T ).name() || typeid( std::uint16_t* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::UINT16;
	}
	else if ( typeid( std::uint32_t ).name() == typeid( T ).name() || typeid( std::uint32_t* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::UINT32;
	}
	else if ( typeid( std::uint64_t ).name() == typeid( T ).name() || typeid( std::uint64_t* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::UINT64;
	}
	else if ( typeid( std::int8_t ).name() == typeid( T ).name() || typeid( std::int8_t* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::INT8;
	}
	else if ( typeid( std::int16_t ).name() == typeid( T ).name() || typeid( std::int16_t* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::INT16;
	}
	else if ( typeid( std::int32_t ).name() == typeid( T ).name() || typeid( std::int32_t* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::INT32;
	}
	else if ( typeid( std::int64_t ).name() == typeid( T ).name() || typeid( std::int64_t* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::INT64;
	}
	else if ( typeid( float ).name() == typeid( T ).name() || typeid( float* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::FLOAT32;
	}
	else if ( typeid( double ).name() == typeid( T ).name() || typeid( double* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::FLOAT64;
	}
	else if ( typeid( char ).name() == typeid( T ).name() || typeid( char* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::CHAR;
	}
	else if ( typeid( bool ).name() == typeid( T ).name() || typeid( bool* ).name() == typeid( T ).name() )
	{
		return Parameters::NUMTYPE::BOOLEAN;
	}
	else
	{
		return Parameters::NUMTYPE::VOID;
	}
}


Parameters::Parameters(const size_t number_of_parameters, const size_t data_size, const std::uint8_t* data)
{
	list.resize( number_of_parameters );
	const size_t parameter_names_size = data_size - 5 * number_of_parameters;

	size_t offset = 0;
	Parameters::NUMTYPE* numtypes = ( Parameters::NUMTYPE* ) ( data + parameter_names_size );
	std::uint32_t* values = ( std::uint32_t*) (data + parameter_names_size + number_of_parameters);
	for ( size_t i = 0; i < number_of_parameters; ++i )
	{
		list.at( i ).name = (char*) ( data + offset );
		list.at( i ).value = num2str(numtypes[i], values[ i ]);
		offset += 1 + list.at(i).name.size();		
	}
}