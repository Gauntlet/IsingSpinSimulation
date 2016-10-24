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

		
		void move_data( JaggedListManager<elem_type>&& that );

		
		void move_data( JaggedList<elem_type>&& that );

		class JAGGED_LIST_MANAGER_SET;

	public:
		JAGGED_LIST_MANAGER_SET set;

		
		JaggedListManager( const uint32_t N, const uint32_t* lengths );

		
		~JaggedListManager();

		/**
		* Deleted copy constructor.
		*/
		JaggedListManager( JaggedListManager<elem_type> const & ) = delete;

		/**
		* Deleted copy operator.
		*/
		JaggedListManager& operator=( JaggedListManager<elem_type> const & ) = delete;


		JaggedListManager( JaggedListManager<elem_type>&& that );
		JaggedListManager( JaggedList<elem_type>&& that );

		
		friend JaggedListManager<elem_type> operator=( JaggedListManager<elem_type>&& that );
		friend JaggedListManager<elem_type> operator=( JaggedList<elem_type>&& that );

		JaggedList<elem_type>& host();
		JaggedList<elem_type>& device();

	};

	template <class elem_type>
	class JaggedListManager<elem_type>::JAGGED_LIST_MANAGER_SET
	{
		JaggedListManager<elem_type>& parent;
	public:
		JAGGED_LIST_MANAGER_SET( JaggedListManager<elem_type>& parent ) : parent( parent ) {};

		
		void host2device();
		void device2host();

		void clear();
	};
}

#endif