#ifndef JAGGED_LIST_SHARED_H
#define JAGGED_LIST_SHARED_H

#include "JaggedList.h"

namespace kspace
{
	/**
	* A container to easily manage JaggedLists that need to be accessed on both the device and host.
	* Updates are controlled by the user of the class using the host2device and device2host functions.
	*/
	template <class elem_type> 
	class JaggedListManager
	{
	private:

		JaggedList<elem_type>* host_ptr;			/**< A pointer to a JaggedList on the host which manages acquired resources on the host*/
		JaggedList<elem_type>* intermediary_ptr;	/**< A pointer to a JaggedList on the host which manages acquired data on the device*/
		JaggedList<elem_type>* device_ptr;			/**< A pointer to a JaggedList which provides access to the resources managed by the intermediary JaggedList*/

		/**
		* Moves data from the passed JaggedListManager to the one that called the function.
		* @param that a JaggedListManager object.
		*/
		void move_data( JaggedListManager<elem_type>&& that );

		/**
		* Moves the resources managed by the JaggedList passed to be managed by one created by the JaggedListManager container
		* @param a JaggedList object.
		*/
		void move_data( JaggedList<elem_type>&& that );

		class JAGGED_LIST_MANAGER_SET;

	public:
		JAGGED_LIST_MANAGER_SET set;

		/**
		* Creates and manages JaggedLists on the device and host. The values of the elements on the device and host are the same.
		* @param N the number of lists.
		* @param lengths an array containing the lengths of the lists.
		*/
		JaggedListManager( const uint32_t N, const uint32_t* lengths );

		/**
		* Frees the resources managed by the JaggeListManaged container.
		*/
		~JaggedListManager();

		/**
		* Delete copy constructor.
		*/
		JaggedListManager( JaggedListManager<elem_type> const & ) = delete;

		/**
		* Delete copy operator.
		*/
		JaggedListManager& operator=( JaggedListManager<elem_type> const & ) = delete;

		/**
		* Moves the resources managed by the passed JaggedListManager to be managed by the one being constructed.
		*/
		JaggedListManager( JaggedListManager<elem_type>&& that ) { move_data( that ); }

		/**
		* Moves the resources managed by the JaggedListManager on the RHS to be managed by the one on the LHS
		*/
		JaggedListManager<elem_type> operator=( JaggedListManager<elem_type>&& that )
		{ 
			move_data( that );
			return *this;
		}

		/**
		* Moves the resources managed by the passed JaggedList to be managed by the JaggedListManager one being constructed.
		*/
		JaggedListManager( JaggedList<elem_type>&& that ) { move_data( that ); }

		/**
		* Moves the resources managed by the JaggedList on the RHS to be managed by the JaggedListManager one on the LHS
		*/
		JaggedListManager<elem_type> operator=( JaggedList<elem_type>&& that )
		{
			move_data( that );
			return *this;
		}

		/**
		* Provides functions on the host acccess to the JaggedList.
		* @return reference to the JaggedList.
		*/
		JaggedList<elem_type>& host() { return *host_ptr; }

		/**
		* Provides functions on the device acccess to the JaggedList.
		* @return reference to the JaggedList.
		*/
		JaggedList<elem_type>& device() { return *device_ptr; }

	};

	template <class elem_type>
	class JaggedListManager<elem_type>::JAGGED_LIST_MANAGER_SET
	{
		JaggedListManager<elem_type>& parent;
	public:
		JAGGED_LIST_MANAGER_SET( JaggedListManager<elem_type>& parent ) : parent( parent ) {};

		/**
		* Copies the data to the device from the host.
		*/
		void host2device();

		/**
		* Copies the data to the host from the device.
		*/
		void device2host();

		/**
		* Frees the resources managed by this class.
		*/
		void clear();
	};
}

#endif