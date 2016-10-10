#ifndef Array_Handle_H
#define Array_Handle_H

#include <stdexcept>
#include "DataStructures.h"

namespace kspace
{
	template<class elem_type>
	class ArrayHandle : public unique_ptr < elem_type[] >
	{
	private:
		MemoryLocation memloc;
		elem_type* data_ptr;
		size_t* number_of_elements;
	protected:
		template <class T>
		class GET
		{
			ArrayHandle<T> const & parent;
		public:
			GET( ArrayHandle<T> const & parent ) : parent(parent) {};

			size_t const & size() { return parent.number_of_elements; }
			elem_type const & operator()( const size_t i )
			{
				if ( size() <= i )
				{
					throw std::out_of_range( "ArrayHandle index is out of bounds" );
				}

				return parent.data_ptr[ i ];
			}

			elem_type const * data() { return parent.data_ptr; }
		};

		template<class T>
		class SET
		{
			ArrayHandle<T>& parent;
		public:
			SET( ArrayHandle<T> const & parent ) : parent( parent ) {};

			elem_type & operator()( const size_t i )
			{
				if ( size() <= i )
				{
					throw std::out_of_range( "ArrayHandle index is out of bounds" );
				}

				return parent.data_ptr[ i ];
			}

			elem_type* data() { return parent.data_ptr; }
		};
	public:
		GET<elem_type> get;
		SET<elem_type> set;

		ArrayHandle( const size_t number_of_elements ) : number_of_elements( new size_t( number_of_elements ) ), data_ptr( new elem_type[ number_of_elements ]() ), memloc(MemoryLocation::host), get(*this), set(*this) {};

		ArrayHandle(const size_t number_of_elements, const MemoryLocation memloc) : number_of_elements(new size_t(number_of_elements)), memloc(memloc), get(*this), set(*this)
		{
			if ( MemoryLocation::host == memloc )
			{
				data_ptr = new elem_type[ number_of_elements ]();
			}
			else if ( MemoryLocation::device == memloc )
			{
				HANDLE_ERROR( cudaMalloc( (void**) &data_ptr, sizeof( elem_type )*number_of_elements ) );
			}
		};


		//Delete copy constructor and assignment operator
		ArrayHandle( const ArrayHandle<elem_type>& ) = delete;
		ArrayHandle& operator=( const ArrayHandle<elem_type>& ) = delete;


		//Define move constructor and assignment operator
		ArrayHandle( ArrayHandle<elem_type>&& that ) : number_of_elements( std::move( that.number_of_elements ) ), unique_ptr<elem_type[]>( std::move( ( unique_ptr<elem_type[]>&& ) that ) )
		{
			that.number_of_elements = 0;
		};

		ArrayHandle& operator=( ArrayHandle<elem_type>&& that )
		{
			number_of_elements = std::move( that.number_of_elements );
			memloc = std::move( that.memloc );

			that.number_of_elements = nullptr;
			that.data_ptr = nullptr;
		}
	};

	void foo()
	{
		ArrayHandle<int> ah( 10 );

		for ( size_t i = 0; i < ah.get.size(); i++ )
		{
			ah.set( i ) = ( i + 1 ) * 10;
		}
	}
}

#endif