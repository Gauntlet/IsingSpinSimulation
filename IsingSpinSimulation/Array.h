#ifndef Array_Handle_H
#define Array_Handle_H

#include <stdexcept>
#include "details.h"

namespace kspace
{

	/**
	* Array template class.
	* Manages the memory of an array of data whether it is stored on the
	* host or device memory. Array objects can not be copied only moved
	* to ensure that memory is correctly freed once out of scope.
	* Array memory is not dynamic and thus not resizable.
	*/
	template <class elem_type>
	class Array
	{
	private:
		MemoryLocation memloc;		/**< MemoryLocation indicates where the data is stored. */
		elem_type* data_ptr;		/**< elem_type pointer to the data. */
		unsigned long int* number_of_elements;	/**< size_t pointer to the number of elements. */
	
		void move_data( Array<elem_type>&& that );

	protected:
		class ARRAY_GET;
		class ARRAY_SET;

	public:	
		Array();
		Array( const size_t number_of_elements );
		Array( const size_t number_of_elements, const MemoryLocation memloc );

		~Array();

		/**
		* Deleted the copy constructor.
		* Array objects can not be copied. In future copy capability 
		* may be introduced through a public method if necessary.
		*/
		Array( const Array<elem_type>& ) = delete;

		/**
		* Deleted the copy assignment operator.
		*/
		Array<elem_type>& operator=( const Array<elem_type>& ) = delete;

		/**
		* Move constructor.
		* Moves the pointers stored in 'that' Array to be managed by the one being constructed..
		* @param an Array object.
		*/
		Array( Array<elem_type>&& that )
		{
			move_data( that );
		};

		/**
		* Move assignment operator.
		* Moves pointers stored in the Array on the RHS to be managed by the one on the LHS.
		* @param an Array object on the RHS.
		* @return an A
		*/
		Array<elem_type>& operator=( Array<elem_type>&& that )
		{
			move_data( that );
			return *this;
		}

		/**
		* get is a public object that provides methods with read only access to private data stored within Array.
		*/
		ARRAY_GET get;

		/**
		* set is a public object that provides methods with read and write access to private data stored within Array..
		*/
		ARRAY_SET set;
	};

	/**
	* A ARRAY_GET class for the Array class.
	* This class contains functions which allow read only access to data stored by the Array class.
	*/
	template<class elem_type>
	class Array<elem_type>::ARRAY_GET
	{
		Array<elem_type> const & parent; /**< A const reference to an Array<elem_type> object. */
	public:
		/**
		* Parameterised Constructor.
		* Takes a parent object whose private and protected members are exposed to the ARRAY_GET class.
		* @param parent a const reference to an Array<elem_type> object.
		*/
		ARRAY_GET( Array<elem_type> const & parent ) : parent( parent ) {};

		
		MemoryLocation const & memory_location() const;
		
		unsigned long int const & size() const;
		elem_type const & operator()( const size_t i ) const;
		elem_type const * data() const;

	};

	/**
	* A ARRAY_SET class for the Array class.
	* This class contains functions which allow read and write access to data stored by the Array class.
	*/
	template<class elem_type>
	class Array<elem_type>::ARRAY_SET
	{
		Array<elem_type> const & parent; /**< A reference to an Array<elem_type> object. */
	public:
		/**
		* A Constructor.
		* Takes a parent object whose private and protected members are exposed to the ARRAY_SET class.
		* @param parent a const reference to an Array<elem_type> object.
		*/
		ARRAY_SET( Array<elem_type> const & parent ) : parent( parent ) {};

		
		elem_type & operator()( const size_t i );
		elem_type* data() const;

		
		elem_type*& data_ptr() const;
		unsigned long int& size() const;

		
		void clear();
	};
}

#endif