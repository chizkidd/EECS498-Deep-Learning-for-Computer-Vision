"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # Hint: X.grad.data stores the gradients                                     #
    ##############################################################################
    # # Forward pass
    # scores = model(X)

    # # Select the correct class scores and sum over batch
    # correct_scores = scores.gather(1, y.view(-1, 1)).sum()

    # # Backward pass
    # model.zero_grad()
    # correct_scores.backward()

    # # Saliency is the max magnitude across color channels
    # saliency = X.grad.data.abs().max(dim=1)[0]
    
    # Forward pass: get scores for all classes
    scores = model(X)
    
    # Get the scores for the correct class (y)
    # We use gather to select the score corresponding to label y for each image
    correct_scores = scores.gather(1, y.view(-1, 1)).squeeze()
    
    # Backward pass: compute gradient of sum of correct scores w.r.t. X
    # Summing avoids needing to pass a gradient tensor to backward()
    correct_scores.sum().backward()
    
    # Saliency is the max absolute value of the gradient across color channels
    # X.grad shape is (N, 3, H, W) -> we want (N, H, W)
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate an adversarial attack X_adv that the model will classify    #
    # as the class target_y. You should perform gradient ascent on the score     #
    # of the target class, stopping when the model is fooled.                    #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate an adversarial     #
    # attack in fewer than 100 iterations of gradient ascent.                    #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    for i in range(max_iter):
        scores = model(X_adv)
        pred = scores.argmax(dim=1).item()

        # Get the current predicted class and the scores we need for printing
        max_score, p_idx = torch.max(scores, dim=1)
        max_score = max_score[0]
        target_score = scores[0, target_y]
        
        if pred == target_y:
            if verbose:
                print(f"\nAttack succeeded at iteration {i} with a target score of {target_score:.3f} that matches the max score of {max_score:.3f}")
            break
            
        # Compute gradient of target class WRT image pixels
        target_score.backward()
        
        # Normalize gradient and update image
        with torch.no_grad():
          grad = X_adv.grad
          if grad.norm() > 0:
            X_adv.data += learning_rate * grad / grad.norm()
        
        # Zero gradients
        X_adv.grad.zero_()
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # # after each step.                                                     #
    # ########################################################################
    # # Forward pass
    # scores = model(img)
    
    # # Backward pass for target class
    # target_score = scores[0, target_y]
    # target_score.backward()
    
    # # Get gradient with L2 regularization
    # grad = img.grad.data - 2 * l2_reg * img.data
    
    # # Update image
    # img.data += learning_rate * grad / grad.norm()
    
    # # Zero gradients
    # img.grad.zero_()

    #------------------------------------------------------------------------
    # Ensure the image tracks gradients
    img.requires_grad_()

    # 1. Forward pass: calculate class scores
    scores = model(img)
    target_score = scores[0, target_y]
    
    # 2. Define the objective: Maximize (Score - L2 Regularization)
    # The L2 term encourages the pixels to stay near zero (natural look)
    l2_penalty = l2_reg * torch.sum(img**2)
    loss = target_score - l2_penalty
    
    # 3. Backward pass to compute d(loss)/d(img)
    loss.backward()
    
    # 4. Gradient Ascent Update
    with torch.no_grad():
        # Retrieve the gradient
        grad = img.grad
        
        # Robust Normalization: prevents steps from being too large or too small
        # 1e-8 prevents division by zero if the gradient is exactly zero
        grad = grad / (grad.norm() + 1e-8)
        
        # Update image pixels directly using += for Gradient Ascent
        img += learning_rate * grad

        # 5. Manually zero the gradient for the next iteration
        # If omitted, next step's gradient would be added to this one
        img.grad.zero_()

    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
