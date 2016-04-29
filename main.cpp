#define PROCESSING_TYPE None
#include "Poisson.hpp" 
#include "Essentials/Exporter.hpp"

int main(){
    brahand::ExporterPointer exporterPtr = VTK_EXPORTER;
    brahand::ImageSize size{100, 50}; 
    float radius = 5;
    float separationWidth = 0.04;
    float separationHeight = 0.08;

    brahand::IndicesArray coordinates = brahand::IndicesArray(size.total());
    brahand::TracePropagationArray propagation = brahand::TracePropagationArray(size.total());
    for(brahand::ulsize i = 0 ; i < size.total() ; ++i){ 
        coordinates[i] = i;
        propagation[i].density = radius;
    }

    brahand::FastMarchingPropagation density{propagation, (brahand::CoordinateDistance)radius, (brahand::CoordinateDistance)radius};
	poisson::PoissonSampler sampler(size, radius, 30, coordinates, density);
    brahand::IndicesArray samples = sampler.sampling();
	brahand::IndicesArray margins = poisson::addMarginPoints(size,separationWidth,separationHeight); 
	
    exporterPtr->exportCoordinates(coordinates, size, "/Users/Pame/Desktop/", "coordinates");
    exporterPtr->exportCoordinates(samples, size, "/Users/Pame/Desktop/", "samples");
	exporterPtr->exportCoordinates(margins, size, "/Users/Pame/Desktop/", "margins”);
    
    return 0;
}
