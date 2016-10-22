#include <sstream>
#include "Parameters.h"
#include "details.h"
using namespace kspace::GRAPH;


/**
* This casts a basic type variable into another basic type without changing the bits.
*
* For example:
*
* float a = 4.3;	//A float.
*
* int b = preserved_cast<int>(a);	//a has been casted into an int such that no data is lost.
*
* float c = preserved_cast<float>(b); //b has been casted back into a float and we obtain the original float. c == 4.3.
*
* int d = (int) a; //a has been casted into an int, the .3 is lost.
* float f = (float) d; //d is cast back itno a float but we no longer have the .3. f==4.
*/
template <class RT, class T>
RT preserved_cast( T num )
{
	return *( (RT*) ( &num ) );
}

/**
* Recasts a pointer into another type of pointer.
*
* Allows the ability to write and read multibyte numbers from an array of bytes.
*/
template <class RT, class T>
RT preserved_cast( T* num )
{
	return *( (RT*) ( num ) );
}

/**
* The values of parameters are stored in a byte array in the file and consist of many different types.
* To store them in a list and be able to access them easily we store them as string. This function will
* cast to the string after being cast to the appropriate type.
*/
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

/**
* When parameter values are stored in the file they are stored as a series of bytes. To identify how these values should be read
* when loading from file the type the value is stored as well. This function returns an enum which indicates the type of the value
* to it.
*/
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

/**
* Returns an enum depending on the template type.
* 
* When parameter values are stored in the file they are stored as a series of bytes, to identify the type of the value
* an enum indicating the type is also stored. This function provides the enum based on the template typename passed to it.
*/
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

/**
* Converts a string into a number.
*
* Parameter values are stored as strings in the list. Numerical parameters are converted back into a number before
* being written to file as this saves space.
*/
template <class T>
T str2num( const std::string strnum )
{
	T num;
	std::stringstream ss;
	ss << strnum;
	ss >> num;
	return num;
}

/**
* Parameterised constuctor that creates a list of parameters containing the standard parameters.
*/
Parameters::Parameters( const Graph::ID graph_id, const std::int32_t number_of_nodes, const std::uint32_t number_of_edges ) : get( *this ), set( *this )
{
	list.resize( 3 );
	list.at( 0 ) = Parameter( "GRAPH_ID", sizeof( Graph::ID ), num2str( Parameters::NUMTYPE::UINT8, graph_id) );
	list.at( 1 ) = Parameter( "NUMBER_OF_NODES", sizeof( std::int32_t ), num2str( Parameters::NUMTYPE::INT32, number_of_nodes ) );
	list.at( 2 ) = Parameter( "NUMBER_OF_EDGES", sizeof( std::uint32_t ), num2str( Parameters::NUMTYPE::UINT32, number_of_edges ) );
}

/**
* Constructs a list of parameters from the parameters passed in the file format.
*/
Parameters::Parameters( const FileFormattedParameters& params ) : get( *this ), set( *this )
{
	list.resize( params.num_of_parameters );
	list.at( 0 ) = Parameter( "GRAPH_ID", sizeof( Graph::ID ), num2str( Parameters::NUMTYPE::CHAR, params.id ) );
	list.at( 1 ) = Parameter( "NUMBER_OF_NODES", sizeof( std::int32_t ), num2str( Parameters::NUMTYPE::INT32, params.num_of_nodes ) );
	list.at( 2 ) = Parameter( "NUMBER_OF_EDGES", sizeof( std::uint32_t ), num2str( Parameters::NUMTYPE::UINT32, params.num_of_edges ) );

	size_t names_size = 0;
	for ( size_t i = 0; i < params.num_of_parameters - 3; ++i )
	{
		list.at( 3 + i ).value = (char*) params.parameters.set.data_ptr() + names_size;
		names_size += list.at( 3 + i ).value.size() + 1;
	}


	size_t values_offset = 0;
	for ( size_t i = 0; i < params.num_of_parameters - 3; ++i )
	{
		list.at( 3 + i ).vtype = ( Parameters::NUMTYPE ) params.parameters.get.data_ptr()[ names_size + i ];

		list.at( 3 + i ).vsize = params.parameters.get.data_ptr()[ names_size + params.num_of_parameters - 3 + i ];

		list.at( 3 + i ).value = num2str( list.at( 3 + i ).vtype, params.parameters.get.data_ptr() + names_size + 2 * params.num_of_parameters - 6 + values_offset );
		values_offset += list.at( 3 + i ).vsize;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* Sets the name, value, value size and value type of a parameter on creation.
*/
Parameters::Parameter::Parameter( const std::string name, const std::uint8_t vsize, const std::string value ) : name( name ), vsize( vsize ), value( value ), vtype( to_numtype( value ) )
{
	
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* A const iterator to the beginning of the parameter list.
*/
std::vector<Parameters::Parameter>::const_iterator Parameters::PARAMETERS_GET::cbegin() const
{
	return parent.list.cbegin();
}

/**
* A const iterator to the end of the parameter list.
*/
std::vector<Parameters::Parameter>::const_iterator Parameters::PARAMETERS_GET::cend() const
{
	return parent.list.cend();
}

/**
* The number of parameters in the list.
*/
std::size_t Parameters::PARAMETERS_GET::size() const
{
	return parent.list.size();
}

/**
* Read only access to a parameter at a particular position in the list.
*/
Parameters::Parameter const & Parameters::PARAMETERS_GET::at( const std::size_t p ) const
{
	return parent.list.at( p );
}

/**
* Formats the the list of parameters into a byte array which can be written into kgraph file.
*/
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

/**
* An iterator to the beginning of the parameter list.
*/
std::vector<Parameters::Parameter>::iterator Parameters::PARAMETERS_SET::begin()
{
	return parent.list.begin();
}

/**
* An iterator to the end of the parameter list.
*/
std::vector<Parameters::Parameter>::iterator Parameters::PARAMETERS_SET::end()
{
	return parent.list.end();
}

/**
* Adds a parameter to the list.
*/
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

/**
* Removes the named parameter from the list.
*/
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

/**
* Removes the parameter at the indicated position from the list.
*/
void Parameters::PARAMETERS_SET::remove( const std::size_t index )
{
	parent.list.erase( begin() + index );
}

/**
* Provides read and write access to a particular parameter in the list.
*/
Parameters::Parameter& Parameters::PARAMETERS_SET::at( const std::size_t p )
{
	return parent.list.at( p );
}