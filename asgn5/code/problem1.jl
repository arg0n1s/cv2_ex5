using Images
using PyPlot
using LightGraphs
#using GraphPlot
using GaussianMixtures

function fit_colors(img, fgmask, k)

end

function data_term(img, fgm, bgm, s, t)
end

function contrast_weights(img, edges)
  number_of_edges = size(edges,1)
  beta = 0
  weights = zeros(number_of_edges)
  for i in 1:number_of_edges
    beta += (img[edges[i,1]] - img[edges[i,2]])^2
  end

  beta = 1/(beta * 1/number_of_edges)

  for i in 1:number_of_edges
  weights[i] = exp(-beta * (img[edges[i,1]] - img[edges[i,2]]))^2
  end

  return weights
end

function make_edges(h,w)
  edges = Vector{Array{Int64,2}}()
  @assert h > 1 && w > 1
  s = (h,w)
  for j in 1:w
    for i in 1:h
      if j < w
        push!(edges, [sub2ind(s,i,j) sub2ind(s,i,j+1)])
      end
      if i < h
        push!(edges, [sub2ind(s,i,j) sub2ind(s,i+1,j)])
      end
    end
  end
  E = edges[1]
  E = map(e -> E = vcat(E, e), edges[2:end])[end]
  return E
end

function make_graph(h,w)
  E = make_edges(h,w)
  number_of_nodes = h*w
  number_of_nodes += 1
  source_index = number_of_nodes
  number_of_nodes += 1
  target_index = number_of_nodes
  #=for i in 1:number_of_nodes-2
    E = vcat(E, [source_index i])
    E = vcat(E, [target_index i])
  end=#
  G = Graph(number_of_nodes-2)
  # = Graph(number_of_nodes)
  for i in 1:size(E,1)
    add_edge!(G, E[i,1], E[i,2])
  end
  return G, E, s, t
end

function smoothness_term(edges, W, lambda, hw)
  smoothness_weights = lambda * W
  return sparse(edges[:,1], edges[:,2], smoothness_weights, hw, hw)
end

function iterated_graphcut(img, bbox, lambda, k)
end



#imshow(iterated_graphcut(img, bbox, 10, 5);
#title("Final Segmentation");
