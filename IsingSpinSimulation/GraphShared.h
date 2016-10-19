#ifndef GRAPH_SHARED_H
#define GRAPH_SHARED_H

#include "DataStructures.h"
#include "Graph.h"

namespace kspace
{
	namespace GRAPH
	{
		class GraphShared
		{
		private:
			Graph* host_ptr;
			Graph* intermediary_ptr;
			Graph* device_ptr;
		protected:

			class GET
			{
			private:
				GraphShared const & parent;
			public:
				GET( GraphShared const & parent ) : parent( parent ) {};

				Graph* host_ptr() const
				{
					return parent.host_ptr;
				}

				Graph* device_ptr() const
				{
					return parent.device_ptr;
				}
			};

			class SET
			{
			private:
				GraphShared& parent;
			public:
				SET( GraphShared& parent ) : parent( parent ) {};

				Graph* host_ptr() const
				{
					return parent.host_ptr;
				}

				Graph* device_ptr() const
				{
					return parent.device_ptr;
				}
			};

		public:

			GraphShared( const std::string filename );
			~GraphShared();

			GET get;
			SET set;

			Graph& host();
			Graph& device();
			Graph& intermediary();

			//Remove the default copy constructors
			GraphShared( const GraphShared& ) = delete;
			GraphShared& operator=( const GraphShared& ) = delete;

			//Remove the default move constructors
			GraphShared( GraphShared&& );
			GraphShared& operator=( GraphShared&& );

			void host2device();
			void device2host();
		};
	}
}
#endif