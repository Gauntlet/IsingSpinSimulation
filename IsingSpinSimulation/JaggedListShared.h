#ifndef JAGGED_LIST_SHARED_H
#define JAGGED_LIST_SHARED_H

namespace kspace
{
	template <class elem_type> class JaggedListShared
	{
	private:
		JaggedList<elem_type>* intermediary_ptr;
		JaggedList<elem_type>* host_ptr;
		JaggedList<elem_type>* device_ptr;

	public:
		JaggedList<elem_type>& host() { return *host_ptr; }
		JaggedList<elem_type>& device() { return *device_ptr; }

		JaggedListShared( const uint32_t N, const uint32_t* lengths );
		~JaggedListShared();

		void host2device();
		void device2host();

	};
}

#endif