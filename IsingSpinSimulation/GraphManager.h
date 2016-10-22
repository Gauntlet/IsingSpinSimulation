#ifndef GRAPH_MANAGER_H
#define GRAPH_MANAGER_H

#include "Graph.h"

namespace kspace
{
	namespace GRAPH
	{

		/**
		* Manages a graph such that is available on device and host processes.
		* Updates of any changes must be done manually by calling set.host2device or set.device2host.
		*/
		class GraphManager
		{
		private:
			Graph* host_ptr;
			Graph* intermediary_ptr;
			Graph* device_ptr;

			void move_data( GraphManager&& );
		protected:

			/**
			* Contains methods which provide read only access to the managed graphs.
			*/
			class GRAPH_MANAGER_GET
			{
			private:
				GraphManager const & parent;
			public:
				GRAPH_MANAGER_GET( GraphManager const & parent ) : parent( parent ) {};

				Graph const * host_ptr() const;
				Graph const * device_ptr() const;

				Graph const & host();
				Graph const & device();
				Graph const & intermediary();
			};

			/**
			* Contains methods which provide read and write access to the managed graphs.
			*/
			class GRAPH_MANAGER_SET
			{
			private:
				GraphManager& parent;
			public:
				GRAPH_MANAGER_SET( GraphManager& parent ) : parent( parent ) {};

				Graph* host_ptr() const;
				Graph* device_ptr() const;

				void host2device();
				void device2host();

				void clear();
			};

		public:
			GRAPH_MANAGER_GET get;
			GRAPH_MANAGER_SET set;

			GraphManager( const std::string filename );
			~GraphManager();

			GraphManager( const GraphManager& ) = delete;
			GraphManager& operator=( const GraphManager& ) = delete;

			GraphManager( GraphManager&& that ) : get( *this ), set( *this )
			{
				move_data( std::move(that) );
			}

			GraphManager& operator=( GraphManager&& that)
			{
				move_data( std::move(that) );
				return *this;
			}

			Graph& host();
			Graph& device();
			Graph& intermediary();
		};
	}
}
#endif