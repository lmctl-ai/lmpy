#include "host.hpp"

namespace lmnum {

void register_families(Api& api) {
  register_sparse(api);
  register_stats(api);
  register_optimize(api);
  register_ode(api);
  register_interp(api);
  register_signal(api);
  register_spatial(api);
  register_graph(api);
  register_models(api);
  register_ml(api);
}

}  // namespace lmnum
