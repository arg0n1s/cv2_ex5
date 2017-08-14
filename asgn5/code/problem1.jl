using Images
using PyPlot
using LightGraphs
using GaussianMixtures
using JLD

#Authors: Nicolas Acero, Sebastian Ehmes

function load_images()
    img = PyPlot.imread("../data/img_1.jpg")
    img = convert(Array{Float64,3}, img)
    return img
end

function display_bounding_box(img, bbox, thickness)
  img_box = copy(img)
  img_box[bbox[1]:bbox[1]+thickness, bbox[3]:bbox[3]+bbox[4], 1] = 255.0
  img_box[bbox[1]+bbox[2]-thickness:bbox[1]+bbox[2], bbox[3]:bbox[3]+bbox[4], 1] = 255.0
  img_box[bbox[1]:bbox[1]+bbox[2], bbox[3]:bbox[3]+thickness, 1] = 255.0
  img_box[bbox[1]:bbox[1]+bbox[2], bbox[3]+bbox[4]-thickness:bbox[3]+bbox[4], 1] = 255.0
  figure()
  axis("off")
  title("Initial foreground annotation")
  imshow(1/maximum(img_box) * img_box)
end

function fit_colors(img, fgmask, k)
  fg_pxl_r = img[:,:,1]
  fg_pxl_g = img[:,:,2]
  fg_pxl_b = img[:,:,3]
  bg_pxl_r = img[:,:,1]
  bg_pxl_g = img[:,:,2]
  bg_pxl_b = img[:,:,3]
  fg_pxl=[[fg_pxl_r[i],fg_pxl_g[i],fg_pxl_b[i]] for i in 1:length(fgmask) if fgmask[i]>0.0]
  bg_pxl=[[bg_pxl_r[i],bg_pxl_g[i],bg_pxl_b[i]] for i in 1:length(fgmask) if fgmask[i]==0.0]
  fg = zeros(Float64, length(fg_pxl), 3)
  for i in 1:length(fg_pxl)
    fg[i,1]=fg_pxl[i][1]
    fg[i,2]=fg_pxl[i][2]
    fg[i,3]=fg_pxl[i][3]
  end
  bg = zeros(Float64, length(bg_pxl), 3)
  for i in 1:length(bg_pxl)
    bg[i,1]=bg_pxl[i][1]
    bg[i,2]=bg_pxl[i][2]
    bg[i,3]=bg_pxl[i][3]
  end
  fg_gmm = GMM(k, fg)
  bg_gmm = GMM(k, bg)
  return [fg_gmm, bg_gmm]
end

function data_term(img, fgm, bgm, s, t)
  h = size(img,1)
  w = size(img,2)
  num_pixels = h*w
  source_weights = zeros(num_pixels)
  target_weights = zeros(num_pixels)
  for i in 1:num_pixels
    r,c = ind2sub((h,w),i)
    target_weights[i] = -avll(fgm, transpose(img[r,c,:][:,:]))
    source_weights[i] = -avll(bgm, transpose(img[r,c,:][:,:]))
  end
  s_and_t = [fill(s,num_pixels); fill(t,num_pixels)]
  pixel_indices = collect(1:num_pixels)
  pixel_indices = [pixel_indices; pixel_indices]
  weights = [source_weights; target_weights]
  return sparse(s_and_t, pixel_indices, weights, num_pixels+2, num_pixels+2)
end

function contrast_weights(img, edges)
  number_of_edges = size(edges,1)
  beta = 0
  weights = zeros(number_of_edges)
  h = size(img,1)
  w = size(img,2)
  for i in 1:number_of_edges
    r1, c1 = ind2sub((h,w),edges[i,1])
    r2, c2 = ind2sub((h,w),edges[i,2])
    beta += sum((img[r1,c1,:] - img[r2,c2,:]).^2)
  end

  beta = 1/(beta * 1/number_of_edges)

  for i in 1:number_of_edges
    r1, c1 = ind2sub((h,w),edges[i,1])
    r2, c2 = ind2sub((h,w),edges[i,2])
    weights[i] = exp(-beta * sum((img[r1,c1,:] - img[r2,c2,:]).^2))
  end

  return weights
end

function make_edges(h,w)
  edges = zeros(Int64, 2*(w-1)*(h-1) + w + h - 2, 2)
  @assert h > 1 && w > 1
  s = (h,w)
  idx = 1
  for j in 1:w
    for i in 1:h
      if j < w
        edges[idx,:] = [sub2ind(s,i,j) sub2ind(s,i,j+1)]
        idx += 1
      end
      if i < h
        edges[idx,:] = [sub2ind(s,i,j) sub2ind(s,i+1,j)]
        idx += 1
      end
    end
  end
  return edges
end

function make_graph(h,w)
  E = make_edges(h,w)
  edges = copy(E)
  number_of_nodes = h*w
  extention = zeros(Int, 2*number_of_nodes,2)
  E = [E; extention]
  px_indices = collect(1:number_of_nodes)
  s = number_of_nodes + 1
  t = s+1
  px_idx = 1
  for i in size(edges,1)+1:2:size(E,1)
    E[i,:]= [s px_idx]
    E[i+1,:] = [px_idx t]
    px_idx += 1
  end
  G = DiGraph(number_of_nodes+2)
  for i in 1:size(E,1)
    add_edge!(G, E[i,1], E[i,2])
  end
  return G, edges, s, t
end

function smoothness_term(edges, W, lambda, hw)
  smoothness_weights = lambda * W
  E = copy(edges)
  E = [E; [E[:,2] E[:,1]]]
  return sparse(E[:,1], E[:,2], [smoothness_weights; smoothness_weights], hw, hw)
end

function iterated_graphcut(img, bbox, lambda, k)
  mask = zeros(Float64, size(img[:,:,1]))
  mask[bbox[1]:bbox[1]+bbox[2],bbox[3]:bbox[3]+bbox[4]].=1.0
  h, w = size(img[:,:,1])
  G, E, source, target = make_graph(h, w)
  weights = contrast_weights(img, E)
  S = smoothness_term(E, weights, lambda, h*w+2)
  #mask = JLD.load("../results/segmentation_at_iteration_9.jld", "seg_9")
  for i in 1:10
    fgm, bgm = fit_colors(img, mask, 5)
    D = data_term(img, fgm, bgm, source, target)
    capacity_matrix = [S[1:end-2,:]; D[end-1:end,:]]
    capacity_matrix[:,end-1:end] = transpose(capacity_matrix[end-1:end,:])
    _, _, labels = maximum_flow(G, source, target, capacity_matrix, algorithm=BoykovKolmogorovAlgorithm())
    mask = reshape(labels[1:end-2],h,w)
    mask[mask .== 2] = 0.0
    mask[mask .== 1] = 1.0
    #JLD.save("../results/segmentation_at_iteration_10.jld", "seg_10", mask)
    figure()
    axis("off")
    title("Segmentation at iteration Nr. $i")
    imshow(mask)
    img_and_seg = zeros(size(img))
    for j in 1:size(mask,2)
      for k in 1:size(mask,1)
        if mask[k,j] == 1.0
          img_and_seg[k,j,2] = 255.0
        else
          img_and_seg[k,j,1] = 255.0
        end
      end
    end
    img_and_seg = 0.6*img + 0.4*img_and_seg
    figure()
    axis("off")
    title("Segmentation at iteration Nr. $i, blended with the test image")
    imshow(1/maximum(img_and_seg) * img_and_seg)
  end
  return mask
end

function problem1()
  img = load_images()
  bbox = [11, 156, 44, 156]
  display_bounding_box(img, bbox, 1)
  k = 5.0
  lamda = 10.0
  seg = iterated_graphcut(img, bbox, lamda, k)
  figure()
  axis("off")
  title("Final Segmentation")
  imshow(seg)
  img_and_seg = zeros(size(img))
  for j in 1:size(seg,2)
    for i in 1:size(seg,1)
      if seg[i,j] == 1.0
        img_and_seg[i,j,2] = 255.0
      else
        img_and_seg[i,j,1] = 255.0
      end
    end
  end
  img_and_seg = 0.6*img + 0.4*img_and_seg
  figure()
  axis("off")
  title("Final segmentation blended with the test image")
  imshow(1/maximum(img_and_seg) * img_and_seg)
end

problem1()
