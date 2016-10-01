#ifndef Array_Handle_H
#define Array_Handle_H

#include <cstdint>
#include <utility>
#include "KDetails.h"
#include <stdexcept>

namespace kspace
{
	class ArrayHandle
	{
	private:
		std::uint8_t* data;
		std::uint64_t size;
	protected:
		class GET : public GET_SUPER < ArrayHandle >
		{
		public:
			GET( const ArrayHandle& parent ) : GET_SUPER( parent ) {};

			const std::uint8_t* data(const size_t i) const
			{
				return parent.data;
			}

			std::uint64_t size() const
			{
				return parent.size;
			}

			template <class T>
			T operator()<T>( const size_t i ) const
			{
				if ( i >= size() )
				{
					throw std::runtime_error( "index out of bounds" );
				}
				return *((T*) (parent.data + i));
			}

			template <class T>
			T& operator[]( const size_t i )
			{
				return *((T*) ( parent.data + i ));
			}
		};

		class SET : public SET_SUPER < ArrayHandle >
		{
		public:
			SET( ArrayHandle& parent ) : SET_SUPER( parent ) {};

			std::uint8_t* data() const
			{
				return parent.data;
			}

			std::uint8_t& operator[]( const size_t i )
			{
				return parent.data[ i ];
			}
		};
	public:


		//Define move constructors for the file pointer.
		ArrayHandle( ArrayHandle&& that ) : data( std::move( that.data ) ), size( std::move( that.size ) ), get( *this )
		{
			that.data = nullptr;
			that.size = 0;
		};

		GET get;

		//Define move operator
		ArrayHandle& operator=( ArrayHandle&& that )
		{
			data = that.data;
			that.data = nullptr;
			that.size = 0;
			return *this;
		}


		//Remove the default copy constructors
		//ArrayHandle( const ArrayHandle& ) = delete;
		//ArrayHandle& operator=( const ArrayHandle& ) = delete;
		ArrayHandle( const ArrayHandle& that ) = delete;
		ArrayHandle& operator=( const ArrayHandle& that ) = delete;


		//Define constructor to generate empty array.
		ArrayHandle( const std::uint64_t size ) : data( new std::uint8_t[ size ]() ), size( size ), get( *this ) {};

		//Define a constructor to make a normal array into a managed array.
		explicit ArrayHandle( std::uint8_t*& data, const std::uint64_t size ) : data( data ), size( size ), get( *this ) { data = nullptr; }

		//Destructor
		~ArrayHandle()
		{
			if ( nullptr != data )
			{
				delete[] data;
			}
		}
	};
}

#endif