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
		size_t* number_of_elements;	/**< size_t pointer to the number of elements. */
	protected:
		class ARRAY_GET;
		class ARRAY_SET;

		void move_data( Array<elem_type>&& that );
	public:
		/**
		* get is a public object that provides methods with read only access to private data stored within Array.
		*/
		ARRAY_GET get;

		/**
		* set is a public object that provides methods with read and write access to private data stored within Array..
		*/
		ARRAY_SET set;

		/**
		* Unparameterised Constructor.
		* Will create an Array object with no data.
		* Data can be moved into this container at some other time after creation.
		*/
		Array();

		/**
		* Parameterised constructor creates an array on host .
		* Initialises an Array object with number_of_elements elements on the host.
		* @param number_of_elements a positive integer.
		*/
		Array( const size_t number_of_elements );

		/**
		* Parameterised constructor creates an array on host .
		* Initialises an Array object with number_of_elements elements on the indicated memory.
		* @param number_of_elements a positive integer.
		* @param memloc an enum with value equal to MemoryLocation::host or MemoryLocation::device.
		*/
		Array( const size_t number_of_elements, const MemoryLocation memloc );

		/**
		* Deconstructor that clears the stores data.
		*/
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

		/**
		* Indicates whether the data is stored on device or host memory.
		* @return an enum.
		*/
		MemoryLocation const & memory_location() const
		{
			return parent.memloc;
		}

		/**
		* @return A const reference to the number of elements in the array.
		*/
		size_t const & size() const;

		/**
		* Read only access to individual elements in the array.
		* If the index 'i' is out of range an exception will be thrown.
		* @param i an integer index of the element being accessed.
		* @return A const reference to an element in the array.
		*/
		elem_type const & operator()( const size_t i ) const;

		/**
		* A pointer to the raw data. The access to the data is read only.
		* @return a const pointer to the raw data array.
		*/
		elem_type const * data_ptr() const;
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

		/**
		* Read and write access to individual elements in the array.
		* If the index 'i' is out of range an exception will be thrown.
		* @param i an integer index of the element being accessed.
		* @return A reference to an element in the array.
		*/
		elem_type & operator()( const size_t i );

		/**
		* A pointer to the raw data. Provides read and write access to the data.
		* @return a pointer to the raw data array.
		*/
		elem_type* data_ptr() const;

		/**
		* Frees the memory used by the data stored by the parent Array object.
		* Uses the memloc variable to free memory on host or device and sets the variables to null.
		*/
		void clear();
	};
}

#endif