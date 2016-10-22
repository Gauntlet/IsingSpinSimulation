#include <sstream>
#include "Parameters.h"
#include "details.h"
using namespace kspace::GRAPH;



template <class RT, class T>
RT preserved_cast( T num )
{
	return *( (RT*) ( &num ) );
}

template <class RT, class T>
RT preserved_cast( T* num )
{
	return *( (RT*) ( num ) );
}

template <class T>
std::string num2str( const Parameters::NUMTYPE numtype, T num )
{
	switch ( numtype )
	{
		case Parameters::NUMTYPE::UINT8:
			return std::to_string( preserved_cast<std::uint8_t>( num ) );
			break;

		case Parameters::NUMTYPE::UINT16:
			return  std::to_string( preserved_cast<std::uint16_t>( num ) );
			break;

		case Parameters::NUMTYPE::UINT32:
			return  std::to_string( preserved_cast<std::uint32_t>( num ) );
			break;

		case Parameters::NUMTYPE::UINT64:
			return std::to_string( preserved_cast<std::uint64_t>( num ) );
			break;

		case Parameters::NUMTYPE::INT8:
			return std::to_string( preserved_cast<std::uint8_t>( num ) );
			break;

		case Parameters::NUMTYPE::INT16:
			return std::to_string( preserved_cast<std::uint16_t>( num ) );
			break;

		case Parameters::NUMTYPE::INT32:
			return std::to_string( preserved_cast<std::uint32_t>( num ) );
			break;

		case Parameters::NUMTYPE::INT64:
			return std::to_string( preserved_cast<std::uint64_t>( num ) );
			break;

		case Parameters::NUMTYPE::FLOAT32:
			return std::to_string( preserved_cast<float>( num ) );
			break;

		case Parameters::NUMTYPE::FLOAT64:
			return std::to_string( preserved_cast<double>( num ) );
			break;

		case Parameters::NUMTYPE::BOOLEAN:
			return std::to_string( preserved_cast<bool>( num ) );
			break;
		default:
			throw std::runtime_error( "No castable type available." );
	}
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

template <class T>
Parameters::NUMTYPE to_numtype()
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

template <class T>
T str2num( const std::string strnum )
{
	T num;
	std::stringstream ss;
	ss << strnum;
	ss >> num;
	return num;
}

Parameters::Parameters( const Graph::ID graph_id, const std::int32_t number_of_nodes, const std::uint32_t number_of_edges ) : get( *this ), set( *this )
{

}

Parameters::Parameters( const std::size_t number_of_parameters, const FileFormattedParameters& params ) : get( *this ), set( *this )
{
	list.resize( number_of_parameters );
	list.at( 0 ) = Parameter( "GRAPH_ID", sizeof( Graph::ID ), num2str( Parameters::NUMTYPE::CHAR, params.id ) );
	list.at( 1 ) = Parameter( "NUMBER_OF_NODES", sizeof( std::int32_t ), num2str( Parameters::NUMTYPE::INT32, params.num_of_nodes ) );
	list.at( 2 ) = Parameter( "NUMBER_OF_EDGES", sizeof( std::uint32_t ), num2str( Parameters::NUMTYPE::UINT32, params.num_of_edges ) );

	size_t names_size = 0;
	for ( size_t i = 0; i < number_of_parameters - 3; ++i )
	{
		list.at( 3 + i ).value = (char*) params.parameters.set.data_ptr() + names_size;
		names_size += list.at( 3 + i ).value.size() + 1;
	}


	size_t values_offset = 0;
	for ( size_t i = 0; i < number_of_parameters - 3; ++i )
	{
		list.at( 3 + i ).vtype = ( Parameters::NUMTYPE ) params.parameters.get.data_ptr()[ names_size + i ];

		list.at( 3 + i ).vsize = params.parameters.get.data_ptr()[ names_size + number_of_parameters - 3 + i ];

		list.at( 3 + i ).value = num2str( list.at( 3 + i ).vtype, params.parameters.get.data_ptr() + names_size + 2 * number_of_parameters - 6 + values_offset );
		values_offset += list.at( 3 + i ).vsize;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

Parameters::Parameter::Parameter( const std::string name, const std::uint8_t vsize, const std::string value ) : name( name ), vsize( vsize ), value( value ), vtype( to_numtype( value ) )
{
	
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Parameters::Parameter>::const_iterator Parameters::PARAMETERS_GET::cbegin() const
{
	return parent.list.cbegin();
}

std::vector<Parameters::Parameter>::const_iterator Parameters::PARAMETERS_GET::cend() const
{
	return parent.list.cend();
}

std::size_t Parameters::PARAMETERS_GET::size() const
{
	return parent.list.size();
}

Parameters::Parameter Parameters::PARAMETERS_GET::at( const std::size_t p ) const
{
	return parent.list.at( p );
}

Parameters::FileFormattedParameters Parameters::PARAMETERS_GET::as_array() const
{
	size_t namesa_size = 0;
	size_t valuesa_size = 0;
	//Skip the first three as they are the graph id, the number of nodes and the number of edges in the graph.
	for ( auto it = cbegin() + 3; it != cend(); ++it )
	{
		namesa_size += it->name.size() + 1;
		valuesa_size += it->vsize;
	}

	Array<std::uint8_t> parama( namesa_size + valuesa_size + ( ( ( *this ).size() - 3 ) * 2 ));

	const size_t vtypea_offset = namesa_size;
	const size_t vsizea_offset = vtypea_offset + size() - 3;
	const size_t valuesa_offset = vsizea_offset + size() - 3;

	size_t namesa_index = 0;
	size_t valuesa_index = 0;

	for ( size_t i = 0; i < size(); ++i )
	{
		auto p = at( 3 + i );
		memcpy( parama.set.data_ptr() + namesa_index, p.name.c_str(), sizeof( char )*( p.name.size() + 1 ) );
		memcpy( parama.set.data_ptr() + vtypea_offset + i, &p.vtype, sizeof( std::uint8_t ) );
		memcpy( parama.set.data_ptr() + vsizea_offset + i, &p.vsize, sizeof( std::uint8_t ) );
		memcpy( parama.set.data_ptr() + valuesa_offset + valuesa_index, &p.value, p.vsize );

		//Add an extra 1 for the '\0' character to indicate the end of the string.
		namesa_index += p.name.size() + 1;
		valuesa_index += p.vsize;
	}

	FileFormattedParameters res;
	res.id = str2num<Graph::ID>( at( 0 ).value );
	res.num_of_nodes = str2num<std::int32_t>( at( 1 ).value );
	res.num_of_edges = str2num<std::uint32_t>( at( 2 ).value );
	res.parameters = std::move( parama );
}

////////////////////////////////////////////////////////////////////////////////////

std::vector<Parameters::Parameter>::iterator Parameters::PARAMETERS_SET::begin()
{
	return parent.list.begin();
}

std::vector<Parameters::Parameter>::iterator Parameters::PARAMETERS_SET::end()
{
	return parent.list.end();
}

template<class T>
void Parameters::PARAMETERS_SET::add( const std::string name, const T value )
{
	Parameter p;
	p.name = name;
	p.vsize = sizeof( T );
	p.vtype = to_numtype( value );
	p.value = std::to_string( value );

	parent.list.push_back( p );
}

void Parameters::PARAMETERS_SET::remove( const std::string name )
{
	for ( auto it = begin(); end() != it; ++it )
	{
		if ( name == it->name )
		{
			parent.list.erase( it );
			break;
		}
	}
}

void Parameters::PARAMETERS_SET::remove( const std::size_t index )
{
	parent.list.erase( begin() + index );
}

Parameters::Parameter& Parameters::PARAMETERS_SET::at( const std::size_t p )
{
	return parent.list.at( p );
}