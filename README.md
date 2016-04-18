
Poisson disk sampling by dart throwing.

```C++

#define PROCESSING_TYPE None
#include "Poisson/Poisson.hpp"

#include "Poisson/Essentials/Exporter.hpp"

int main(){
    brahand::ExporterPointer exporterPtr = VTK_EXPORTER;
    brahand::ImageSize size{100, 50};
    float radius = 5;

    brahand::IndicesArray coordinates = brahand::IndicesArray(size.total());
    brahand::TracePropagationArray propagation = brahand::TracePropagationArray(size.total());
    for(brahand::ulsize i = 0 ; i < size.total() ; ++i){
        coordinates[i] = i;
        propagation[i].index = i;
        propagation[i].density = radius;
    }

    brahand::FastMarchingPropagation density{propagation, (brahand::CoordinateDistance)radius, (brahand::CoordinateDistance)radius};

    poisson::PoissonSampler sampler(size, radius, 30, coordinates, density);
    brahand::IndicesArray samples = sampler.sampling();

    exporterPtr->exportCoordinates(coordinates, size, "/Users/Bryan/Desktop/", "coordinates");
    exporterPtr->exportCoordinates(samples, size, "/Users/Bryan/Desktop/", "samples");

    return 0;
}


```
