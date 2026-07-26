#include <TH1.h>

void get_error_on_mean()
{
    TH1::SetDefaultSumw2();
    
    TH1D *h1 = new TH1D("h", "", 12, 0., 12.);
    
    std::vector<double> y = {0.02420524, 0.11806773, 0.18621135, 0.17989836, 0.14347375, 0.10705863, 0.0827742, 0.062193, 0.04205437, 0.02912202, 0.01664538, 0.00829597};
    std::vector<double> yerr = {0.00504769, 0.01218989, 0.00988437, 0.01260129, 0.0077097, 0.00535946, 0.00464973, 0.00525452, 0.00520773, 0.00390926, 0.00260095, 0.0016055};
    
    for (int ibin = 0; ibin < h1->GetNbinsX(); ibin++)
    {
        h1->SetBinContent(ibin+1, y.at(ibin));
        h1->SetBinError(ibin, yerr.at(ibin));
    }
    
    std::cout << h1->GetMean() << std::endl;
    std::cout << h1->GetStdDev() << std::endl;
    std::cout << h1->GetMeanError() << std::endl;
    std::cout << h1->GetStdDevError() << std::endl;

    h1->Draw();
}