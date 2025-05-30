#include <TH1.h>

void get_error_on_mean()
{
    TH1::SetDefaultSumw2();
    
    TH1D *h1 = new TH1D("h", "", 13, 1.5, 14.5);
    
    std::vector<double> y = {4.76856367e-02, 1.14209768e-01, 2.81131458e-01, 2.36694895e-01, 1.80494789e-01, 8.85684368e-02, 3.44459454e-02, 1.19494202e-02, 3.71235394e-03, 8.54729557e-04, 1.50026519e-04, 9.15169974e-05, 1.06583962e-05};
    std::vector<double> yerr = {1.56153054e-03, 2.30265676e-03, 4.03157183e-03, 3.97665568e-03, 3.50230098e-03, 2.49543264e-03, 1.40236967e-03, 8.94955050e-04, 5.48392576e-04, 1.22025314e-04, 2.00604073e-05, 5.77889816e-05, 2.67607032e-06};
    
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