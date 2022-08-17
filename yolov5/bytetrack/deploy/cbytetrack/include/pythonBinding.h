//YourFile.cpp (compiled into a .dll or .so file)
#include <new> //For std::nothrow
#include "BYTETracker.h"

extern "C"  //Tells the compile to use C-linkage for the next scope.
{
    //Note: The interface this linkage region needs to use C only.  
    void * ByteTrackNew( void )
    {
        // Note: Inside the function body, I can use C++. 
        return new(std::nothrow) BYTETracker;
    }

    //Thanks Chris. 
    void ByteTrackDelete (void *ptr)
    {
         delete ptr; 
    }

    uint32_t ByteTrackUpdate(void *ptr, const float* box, uint32_t nbox, float* out)
    {

        // Note: A downside here is the lack of type safety. 
        // You could always internally(in the C++ library) save a reference to all 
        // pointers created of type MyClass and verify it is an element in that
        //structure. 
        //
        // Per comments with Andre, we should avoid throwing exceptions.  
        try
        {
            const uint32_t* cls = nullptr;
            BYTETracker * ref = reinterpret_cast<BYTETracker *>(ptr);
            return ref->update(box, cls, nbox, out);
        }
        catch(...)
        {
           return -1; //assuming -1 is an error condition. 
        }
    }

} //End C linkage scope.