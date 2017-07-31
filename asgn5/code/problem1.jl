using Images
using PyPlot

function fit_colors(img, fgmask, k)

end

function data_term(img, fgm, bgm, s, t)
end

function contrast_weights(img, edges)
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

end

function smoothness_term(edges, W, lambda, hw)
end

function iterated_graphcut(img, bbox, lambda, k)
end



#imshow(iterated_graphcut(img, bbox, 10, 5);
#title("Final Segmentation");
