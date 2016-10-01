#ifndef K_DETAILS_H
#define K_DETAILS_H

namespace kspace {

	template <class PARENT>
	class GET_SUPER
	{
	protected:
		const PARENT& parent;
	public:
		GET_SUPER( const PARENT& parent ) : parent( parent );
	};

	template <class PARENT>
	class SET_SUPER
	{
	protected:
		PARENT& parent;
	public:
		SET_SUPER( PARENT& parent ) : parent( parent );
	};
}

#endif